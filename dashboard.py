import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
from datetime import datetime

# Set page config
st.set_page_config(page_title="Smart Analytics Dashboard", layout="wide")

# --- UTILITY FUNCTIONS ---

def dynamic_map_columns(df):
    """Fuzzy match columns to standard business terms."""
    cols = df.columns.tolist()
    mapping = {}
    
    # Expanded candidates explicitly splitting Limit, Utilization, and Dates
    candidates = {
        "AM": ["am_name", "owner", "account_manager", "rm_name", "relationship_manager", "assigned_to", "am_email"],
        "Account": ["company", "buyer", "seller", "account_name", "client", "customer"],
        "Limit": ["facility_size", "limit", "approved_limit", "overdraft_limit", "max_balance"],
        "Utilization": ["outstanding_balance", "outstanding", "utilization", "total advanced"],
        "Revenue": ["realised revenue", "booked_revenue", "revenue", "mrr", "arr", "total_fees", "amount", "sales"],
        "Status": ["utilization_status", "status", "stage", "state"],
        "DPD": ["dpd", "overdue_days", "days_past_due"],
        "TransactionDate": ["last_disbursed_date", "disbursed_date", "invoice date", "created date", "date", "updated date"]
    }
    
    for key, choices in candidates.items():
        matches = []
        for choice in choices:
            match, score = process.extractOne(choice, cols)
            if score > 85: # High confidence threshold
                if key == "AM" and "name" in match.lower():
                    matches.insert(0, match) # Put name at the front
                else:
                    matches.append(match)
        
        if matches:
            if key == "AM":
                names = [m for m in matches if "name" in m.lower() or "manager" in m.lower() or "owner" in m.lower()]
                mapping[key] = [names[0]] if names else [matches[0]]
            else:
                mapping[key] = [matches[0]]
            
    return mapping

@st.cache_data
def load_data(file):
    """Safely load CSV or Excel data and strip whitespace from headers."""
    filename = file.name.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        return None

# --- UI LAYOUT ---

st.title("📊 Smart Analytics Dashboard")
st.markdown("Upload **any** Excel or CSV files to analyze Account Limits, Utilization, and Risk.")

# Initialize session state for stored datasets
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

with st.sidebar:
    st.header("Data Source")
    uploaded_files = st.file_uploader("Upload Data Files", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
    
    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            # Only load if not already in session state or if modified
            if file.name not in st.session_state.datasets:
                with st.spinner(f"Loading {file.name}..."):
                    loaded_df = load_data(file)
                    if loaded_df is not None and not loaded_df.empty:
                        st.session_state.datasets[file.name] = loaded_df

    # Allow user to select which dataset to view if multiple are loaded
    selected_dataset = None
    if st.session_state.datasets:
        st.divider()
        st.subheader("Select Dataset to Analyze")
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Active Dataset", dataset_names)
        
        # Optional: Clear data
        if st.button("Clear All Data"):
            st.session_state.datasets = {}
            st.rerun()

if selected_dataset and selected_dataset in st.session_state.datasets:
    df = st.session_state.datasets[selected_dataset]
    mapping = dynamic_map_columns(df)
        
    # --- TABBED INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["🎯 Portfolio Intelligence", "🔍 Smart Auto-Discovery", "📋 Raw Data"])
    
    # TAB 1: Advanced Business Logic focusing on categorization and action
    with tab1:
        st.markdown("### Intelligent Portfolio Overview")
        if not mapping:
            st.info("Could not auto-detect standard business columns (AM, Account, Limit, Utilization). Try the 'Smart Auto-Discovery' tab!")
        else:
            # Setup Filter
            plot_df = df.copy()
            selected_am = "All"
            if "AM" in mapping:
                am_col = mapping["AM"][0]
                am_list = ["All"] + sorted([str(x) for x in df[am_col].dropna().unique()])
                selected_am = st.selectbox("Filter by Account Manager", am_list)
                if selected_am != "All":
                    plot_df = plot_df[plot_df[am_col].astype(str) == selected_am]

            # Pre-calculate numerics safely
            if "Limit" in mapping:
                plot_df[mapping["Limit"][0]] = pd.to_numeric(plot_df[mapping["Limit"][0]], errors='coerce').fillna(0)
            if "Utilization" in mapping:
                plot_df[mapping["Utilization"][0]] = pd.to_numeric(plot_df[mapping["Utilization"][0]], errors='coerce').fillna(0)
            
            # --- CALCULATE ADVANCED METRICS ---
            
            acc_col = mapping.get("Account", [None])[0]
            lim_col = mapping.get("Limit", [None])[0]
            util_col = mapping.get("Utilization", [None])[0]
            stat_col = mapping.get("Status", [None])[0]
            date_col = mapping.get("TransactionDate", [None])[0]
            dpd_col = mapping.get("DPD", [None])[0]
            
            if acc_col and lim_col and util_col:
                # Aggregate at account level to avoid duplicate metric calculations
                account_df = plot_df.groupby(acc_col).agg(
                    Total_Limit=(lim_col, 'sum'),
                    Total_Utilization=(util_col, 'sum')
                ).reset_index()
                
                # Calculate Utilization Percentage Category
                # Avoid division by zero
                account_df['Total_Limit_Safe'] = account_df['Total_Limit'].replace(0, pd.NA)
                account_df['Utilization_Pct'] = (account_df['Total_Utilization'] / account_df['Total_Limit_Safe']) * 100
                account_df['Utilization_Pct'] = account_df['Utilization_Pct'].fillna(0)
                
                # UPDATED RULES: Low < 20%, Medium 20-70%, High > 70%
                def categorize_util(pct):
                    if pct > 70: return 'High (>70%)'
                    elif pct >= 20: return 'Medium (20-70%)'
                    else: return 'Low (<20%)'
                    
                account_df['Utilization_Category'] = account_df['Utilization_Pct'].apply(categorize_util)
                
                # Merge Status if available
                if stat_col:
                    # Get the most common status per account
                    status_df = plot_df.groupby(acc_col)[stat_col].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').reset_index()
                    account_df = account_df.merge(status_df, on=acc_col, how='left')
                else:
                    account_df[stat_col] = 'Unknown'
                    
                # Calculate Inactivity if Date is available
                if date_col:
                    plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors='coerce')
                    date_df = plot_df.groupby(acc_col)[date_col].max().reset_index()
                    date_df.rename(columns={date_col: 'Last_Transaction'}, inplace=True)
                    account_df = account_df.merge(date_df, on=acc_col, how='left')
                    
                    # Assuming today is the max date in the dataset to simulate real-time accurately
                    current_date = plot_df[date_col].max() if not pd.isna(plot_df[date_col].max()) else pd.Timestamp.now()
                    account_df['Days_Inactive'] = (current_date - account_df['Last_Transaction']).dt.days
                    
                    def categorize_inactivity(days):
                        if pd.isna(days): return 'Unknown'
                        if days > 60: return 'Inactive (>60 days)'
                        elif days > 30: return 'Dormant (30-60 days)'
                        else: return 'Active (<30 days)'
                        
                    account_df['Activity_Status'] = account_df['Days_Inactive'].apply(categorize_inactivity)
                else:
                    account_df['Last_Transaction'] = 'N/A'
                    account_df['Activity_Status'] = 'Unknown'
                    account_df['Days_Inactive'] = 0

                # Calculate "Action Required" Logic based on new thresholds
                def needs_action(row):
                    reasons = []
                    if row.get('Utilization_Pct', 0) > 70:
                        reasons.append("High Utilization (>70%)")
                    if str(row.get(stat_col, '')).lower() in ['suspended', 'overdue']:
                        reasons.append("Account Suspended/Overdue")
                    if row.get('Days_Inactive', 0) > 60:
                        reasons.append("Inactive (>60 days)")
                    return " | ".join(reasons) if reasons else "Healthy"
                
                account_df['Action_Required'] = account_df.apply(needs_action, axis=1)
                
                # --- RENDER DASHBOARD ---
                
                # Top KPIs
                total_accounts = len(account_df)
                action_accounts = len(account_df[account_df['Action_Required'] != 'Healthy'])
                high_util_count = len(account_df[account_df['Utilization_Category'] == 'High (>70%)'])
                inactive_count = len(account_df[account_df['Activity_Status'] == 'Inactive (>60 days)'])

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Accounts", total_accounts)
                k2.metric("Requires Action ⚠️", action_accounts, delta_color="inverse")
                k3.metric("High Utilization Accounts", high_util_count)
                k4.metric("Inactive Accounts (>60d)", inactive_count)
                
                st.divider()
                
                # Visualizations Row 1
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Utilization Categories")
                    util_counts = account_df['Utilization_Category'].value_counts().reset_index()
                    util_counts.columns = ['Category', 'Count']
                    fig1 = px.pie(util_counts, names='Category', values='Count', hole=0.4, 
                                  color='Category', 
                                  color_discrete_map={'High (>70%)':'#d62728', 'Medium (20-70%)':'#ff7f0e', 'Low (<20%)':'#2ca02c'})
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    fig1.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                    # Make pie chart interactive
                    util_event = st.plotly_chart(fig1, use_container_width=True, on_select="rerun", selection_mode="points", key="util_pie")
                    
                with c2:
                    st.subheader("Activity Status")
                    act_counts = account_df['Activity_Status'].value_counts().reset_index()
                    act_counts.columns = ['Status', 'Count']
                    fig2 = px.pie(act_counts, names='Status', values='Count', hole=0.4,
                                  color='Status',
                                  color_discrete_map={'Inactive (>60 days)':'#7f7f7f', 'Dormant (30-60 days)':'#ffbb78', 'Active (<30 days)':'#98df8a', 'Unknown':'#c7c7c7'})
                    fig2.update_traces(textposition='inside', textinfo='percent+label')
                    fig2.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                    # Make pie chart interactive
                    act_event = st.plotly_chart(fig2, use_container_width=True, on_select="rerun", selection_mode="points", key="act_pie")

                st.divider()
                
                # --- INTERACTIVE ACCOUNT EXPLORER ---
                st.subheader("📂 Interactive Account Explorer")
                st.write("Click on any slice in the pie charts above to filter the accounts below. If no slice is selected, all accounts are shown.")
                
                # Extract selections from events
                selected_utils = []
                if util_event and "selection" in util_event and "points" in util_event["selection"]:
                    for p in util_event["selection"]["points"]:
                        if "label" in p:
                            selected_utils.append(p["label"])
                        elif "point_index" in p and p["point_index"] < len(util_counts):
                            selected_utils.append(util_counts.iloc[p["point_index"]]["Category"])
                            
                selected_acts = []
                if act_event and "selection" in act_event and "points" in act_event["selection"]:
                    for p in act_event["selection"]["points"]:
                        if "label" in p:
                            selected_acts.append(p["label"])
                        elif "point_index" in p and p["point_index"] < len(act_counts):
                            selected_acts.append(act_counts.iloc[p["point_index"]]["Status"])
                    
                # Also keep the "Requires Action" toggle just in case
                action_filter = st.selectbox("Quick Filter:", ["Show All Accounts", "⚠️ Requires Action Only", "✅ Healthy Only"], index=0)
                
                # Apply Filters
                filtered_df = account_df.copy()
                
                # Apply pie chart filters
                if selected_utils:
                    filtered_df = filtered_df[filtered_df['Utilization_Category'].isin(selected_utils)]
                if selected_acts:
                    filtered_df = filtered_df[filtered_df['Activity_Status'].isin(selected_acts)]
                    
                # Apply action filter
                if action_filter == "⚠️ Requires Action Only":
                    filtered_df = filtered_df[filtered_df['Action_Required'] != 'Healthy']
                elif action_filter == "✅ Healthy Only":
                    filtered_df = filtered_df[filtered_df['Action_Required'] == 'Healthy']

                if not filtered_df.empty:
                    display_cols = [acc_col, 'Total_Limit', 'Total_Utilization', 'Utilization_Pct', 'Utilization_Category', 'Activity_Status', 'Action_Required']
                    if stat_col: display_cols.append(stat_col)
                    
                    # Format the percentage for better display
                    display_df = filtered_df[display_cols].sort_values(by='Utilization_Pct', ascending=False)
                    st.dataframe(display_df, use_container_width=True)
                    st.caption(f"Showing {len(display_df)} accounts based on your selection.")
                else:
                    st.info("No accounts match the selected categories.")
                    
            else:
                st.warning("Insufficient data mapped to perform advanced logic. Need Account, Limit, and Utilization.")

    # TAB 2: Generalized "Any Data" Analysis
    with tab2:
        st.markdown("### Smart Auto-Discovery")
        st.write("This tab automatically analyzes **any** dataset, regardless of the column names. It finds the numeric and categorical columns and lets you build clean charts instantly.")
        
        # Detect data types
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        
        if num_cols and cat_cols:
            col_a, col_b = st.columns(2)
            with col_a:
                cat_choice = st.selectbox("Group By (Category)", cat_cols)
            with col_b:
                num_choice = st.selectbox("Measure (Metric)", num_cols)
            
            agg_method = st.radio("Aggregation", ["Sum", "Average", "Count"], horizontal=True)
            
            if agg_method == "Sum":
                agg_df = df.groupby(cat_choice)[num_choice].sum().reset_index()
            elif agg_method == "Average":
                agg_df = df.groupby(cat_choice)[num_choice].mean().reset_index()
            else:
                agg_df = df.groupby(cat_choice)[num_choice].count().reset_index()
            
            # Get Top 15 and plot
            agg_df = agg_df.nlargest(15, num_choice).sort_values(by=num_choice, ascending=True)
            
            st.subheader(f"Top 15 {cat_choice} by {agg_method} of {num_choice}")
            fig_auto = px.bar(agg_df, x=num_choice, y=cat_choice, orientation='h', color=num_choice, color_continuous_scale="Blues")
            st.plotly_chart(fig_auto, use_container_width=True)
        else:
            st.info("The dataset needs at least one numeric and one text column to auto-generate charts.")

    # TAB 3: Raw Profile
    with tab3:
        st.markdown("### Raw Data Profile")
        st.dataframe(df, use_container_width=True)
else:
    st.info("Waiting for file upload. Use the sidebar to upload a CSV or Excel file.")