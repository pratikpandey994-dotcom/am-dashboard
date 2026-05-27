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
                    if row.get('Utilization_Pct', 0) < 20:
                        reasons.append("Low Utilization (<20%)")
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
                low_util_count = len(account_df[account_df['Utilization_Category'] == 'Low (<20%)'])
                inactive_count = len(account_df[account_df['Activity_Status'] == 'Inactive (>60 days)'])

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Accounts", total_accounts)
                k2.metric("Requires Action ⚠️", action_accounts, delta_color="inverse")
                k3.metric("Low Utilization Accounts", low_util_count)
                k4.metric("Inactive Accounts (>60d)", inactive_count)
                
                st.divider()
                
                # Visualizations Row 1
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Utilization Categories")
                    util_counts = account_df['Utilization_Category'].value_counts().reset_index()
                    util_counts.columns = ['Category', 'Count']
                    # Sort for consistent display
                    order = ['High (>70%)', 'Medium (20-70%)', 'Low (<20%)']
                    util_counts['Category'] = pd.Categorical(util_counts['Category'], categories=order, ordered=True)
                    util_counts = util_counts.sort_values('Category')
                    
                    fig1 = px.bar(util_counts, x='Count', y='Category', orientation='h',
                                  color='Category', 
                                  color_discrete_map={'High (>70%)':'#2ca02c', 'Medium (20-70%)':'#ff7f0e', 'Low (<20%)':'#d62728'},
                                  text='Count')
                    fig1.update_traces(textposition='inside', marker_line_color='white', marker_line_width=1.5)
                    fig1.update_layout(showlegend=False, margin=dict(t=10, b=0, l=0, r=0), yaxis_title=None, xaxis_title="Number of Accounts")
                    # Make chart interactive
                    util_event = st.plotly_chart(fig1, use_container_width=True, on_select="rerun", selection_mode="points", key="util_bar")

                with c2:
                    st.subheader("Activity Status")
                    act_counts = account_df['Activity_Status'].value_counts().reset_index()
                    act_counts.columns = ['Status', 'Count']
                    
                    fig2 = px.bar(act_counts, x='Count', y='Status', orientation='h',
                                  color='Status',
                                  color_discrete_map={'Inactive (>60 days)':'#7f7f7f', 'Dormant (30-60 days)':'#ffbb78', 'Active (<30 days)':'#98df8a', 'Unknown':'#c7c7c7'},
                                  text='Count')
                    fig2.update_traces(textposition='inside', marker_line_color='white', marker_line_width=1.5)
                    fig2.update_layout(showlegend=False, margin=dict(t=10, b=0, l=0, r=0), yaxis_title=None, xaxis_title="Number of Accounts")
                    # Make chart interactive
                    act_event = st.plotly_chart(fig2, use_container_width=True, on_select="rerun", selection_mode="points", key="act_bar")

                st.divider()

                # --- INTERACTIVE ACCOUNT EXPLORER ---
                st.subheader("📂 Interactive Account Explorer")
                st.write("Click on any bar in the charts above to filter the accounts below. If no bar is selected, all accounts are shown.")

                # Extract selections from events
                selected_utils = []
                if util_event and "selection" in util_event and "points" in util_event["selection"]:
                    for p in util_event["selection"]["points"]:
                        if "y" in p:
                            selected_utils.append(p["y"])

                selected_acts = []
                if act_event and "selection" in act_event and "points" in act_event["selection"]:
                    for p in act_event["selection"]["points"]:
                        if "y" in p:
                            selected_acts.append(p["y"])                    
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
        st.write("Automatically analyze and visualize the relationships in your dataset.")
        
        # Detect data types
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        
        if num_cols and cat_cols:
            # Layout for controls
            st.markdown("#### ⚙️ Data Configuration")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                cat_choice = st.selectbox("Group By (Category)", cat_cols)
            with col_b:
                num_choice = st.selectbox("Measure (Metric)", num_cols)
            with col_c:
                agg_method = st.selectbox("Aggregation Method", ["Sum", "Average", "Count", "Max", "Min"])
                
            # Layout for visual controls
            v1, v2, v3 = st.columns(3)
            with v1:
                chart_type = st.selectbox("Chart Type", ["Bar Chart", "Donut Chart", "Line Chart", "Scatter Plot"])
            with v2:
                top_n = st.selectbox("Show Top N Results", [10, 15, 25, 50, "All"])
            with v3:
                sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"])

            # Perform Aggregation
            if agg_method == "Sum":
                agg_df = df.groupby(cat_choice)[num_choice].sum().reset_index()
            elif agg_method == "Average":
                agg_df = df.groupby(cat_choice)[num_choice].mean().reset_index()
            elif agg_method == "Count":
                agg_df = df.groupby(cat_choice)[num_choice].count().reset_index()
            elif agg_method == "Max":
                agg_df = df.groupby(cat_choice)[num_choice].max().reset_index()
            elif agg_method == "Min":
                agg_df = df.groupby(cat_choice)[num_choice].min().reset_index()

            # Dynamic Number Formatter
            def format_num(num):
                if pd.isna(num): return "0"
                if abs(num) >= 1e9: return f"{num/1e9:.2f}B"
                elif abs(num) >= 1e6: return f"{num/1e6:.2f}M"
                elif abs(num) >= 1e3: return f"{num/1e3:.2f}K"
                else: return f"{num:,.2f}"

            # Calculate top level summary KPIs
            total_metric = agg_df[num_choice].sum()
            avg_metric = agg_df[num_choice].mean()
            max_metric = agg_df[num_choice].max()
            
            st.divider()
            
            # Show KPIs
            k1, k2, k3 = st.columns(3)
            k1.metric(f"Total {num_choice}", format_num(total_metric))
            k2.metric(f"Avg {num_choice} per {cat_choice}", format_num(avg_metric))
            k3.metric(f"Max {num_choice}", format_num(max_metric))
            
            st.divider()

            # Title
            title_text = f"Top {top_n} {cat_choice} by {agg_method} of {num_choice}" if top_n != "All" else f"All {cat_choice} by {agg_method} of {num_choice}"
            st.subheader(title_text)

            # Sorting and Limiting (always take top magnitude first for 'descending' logic)
            if top_n != "All":
                agg_df = agg_df.nlargest(top_n, num_choice)
                
            is_ascending = (sort_order == "Ascending")
            agg_df = agg_df.sort_values(by=num_choice, ascending=is_ascending)

            # Draw Chart
            if chart_type == "Bar Chart":
                fig_auto = px.bar(agg_df, x=cat_choice, y=num_choice, color=num_choice, color_continuous_scale="Blues", text_auto='.2s')
                fig_auto.update_layout(xaxis_tickangle=-45)
            elif chart_type == "Donut Chart":
                fig_auto = px.pie(agg_df, names=cat_choice, values=num_choice, hole=0.4)
                fig_auto.update_traces(textposition='inside', textinfo='percent+label')
            elif chart_type == "Line Chart":
                fig_auto = px.line(agg_df, x=cat_choice, y=num_choice, markers=True)
                fig_auto.update_layout(xaxis_tickangle=-45)
            elif chart_type == "Scatter Plot":
                # Ensure size doesn't fail on negative values (e.g. negative DPD)
                safe_size = agg_df[num_choice].apply(lambda x: max(x, 0))
                fig_auto = px.scatter(agg_df, x=cat_choice, y=num_choice, size=safe_size, color=num_choice, color_continuous_scale="Blues")
                fig_auto.update_layout(xaxis_tickangle=-45)
                
            st.plotly_chart(fig_auto, use_container_width=True)
            
            with st.expander("Show Data Table"):
                # Use st.dataframe with pandas styling for formatting
                st.dataframe(agg_df.style.format({num_choice: "{:,.2f}"}), use_container_width=True)

        else:
            st.info("The dataset needs at least one numeric and one text column to auto-generate charts.")

    # TAB 3: Raw Profile
    with tab3:
        st.markdown("### Raw Data Profile")
        st.dataframe(df, use_container_width=True)
else:
    st.info("Waiting for file upload. Use the sidebar to upload a CSV or Excel file.")