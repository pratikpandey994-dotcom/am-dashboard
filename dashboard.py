import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process

# Set page config
st.set_page_config(page_title="Smart Analytics Dashboard", layout="wide")

# --- UTILITY FUNCTIONS ---

def dynamic_map_columns(df):
    """Fuzzy match columns to standard business terms."""
    cols = df.columns.tolist()
    mapping = {}
    
    # Expanded candidates explicitly splitting Limit and Utilization
    candidates = {
        "AM": ["am_name", "owner", "account_manager", "rm_name", "relationship_manager", "assigned_to", "am_email"],
        "Account": ["company", "buyer", "seller", "account_name", "client", "customer"],
        "Limit": ["facility_size", "limit", "approved_limit", "overdraft_limit", "max_balance"],
        "Utilization": ["outstanding_balance", "outstanding", "utilization", "total advanced"],
        "Revenue": ["realised revenue", "booked_revenue", "revenue", "mrr", "arr", "total_fees", "amount", "sales"],
        "Status": ["utilization_status", "status", "stage", "state"],
        "DPD": ["dpd", "overdue_days", "days_past_due"],
        "Date": ["date", "created", "closed", "updated", "disbursed", "settlement", "invoice"]
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
st.markdown("Upload **any** Excel or CSV file to analyze Account Limits, Utilization, and Risk.")

with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Data File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("Could not read the file. Please ensure it's a valid CSV or Excel document with data.")
    else:
        mapping = dynamic_map_columns(df)
        
        # --- TABBED INTERFACE ---
        tab1, tab2, tab3 = st.tabs(["🎯 Limits & Utilization", "🔍 Smart Auto-Discovery", "📋 Raw Data"])
        
        # TAB 1: Business Logic focusing on Limits vs Utilization
        with tab1:
            st.markdown("### Portfolio Overview")
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

                # KPIs
                m1, m2, m3, m4 = st.columns(4)
                
                # Metric 1: Total Accounts
                if "Account" in mapping:
                    acc_col = mapping["Account"][0]
                    m1.metric("Total Unique Accounts", plot_df[acc_col].nunique())
                else:
                    m1.metric("Total Rows", len(plot_df))
                    
                # Metric 2: Total Facility Limit
                total_limit = 0
                if "Limit" in mapping:
                    total_limit = plot_df[mapping["Limit"][0]].sum()
                    m2.metric("Total Facility Limit", f"${total_limit:,.0f}")
                else:
                    m2.metric("Total Facility Limit", "N/A")
                    
                # Metric 3: Total Utilization
                total_utilization = 0
                if "Utilization" in mapping:
                    total_utilization = plot_df[mapping["Utilization"][0]].sum()
                    m3.metric("Total Utilization", f"${total_utilization:,.0f}")
                else:
                    m3.metric("Total Utilization", "N/A")
                
                # Metric 4: Utilization % or Risk
                if total_limit > 0:
                    util_pct = (total_utilization / total_limit) * 100
                    m4.metric("Overall Utilization %", f"{util_pct:.1f}%")
                elif "Status" in mapping:
                    stat_col = mapping["Status"][0]
                    risk_mask = plot_df[stat_col].astype(str).str.contains("suspended|risk|overdue", case=False, na=False)
                    if "Account" in mapping:
                        risk_count = plot_df[risk_mask][acc_col].nunique()
                    else:
                        risk_count = risk_mask.sum()
                    m4.metric("Accounts at Risk", risk_count, delta_color="inverse")
                else:
                    m4.metric("Accounts at Risk", "N/A")

                st.divider()
                
                # Visualizations
                c1, c2 = st.columns([2, 1])
                
                # Chart 1: Limits vs Utilization (Grouped Bar)
                with c1:
                    if "Account" in mapping and "Limit" in mapping and "Utilization" in mapping:
                        st.subheader("Limits vs. Utilization (Top Accounts)")
                        acc_col = mapping["Account"][0]
                        lim_col = mapping["Limit"][0]
                        util_col = mapping["Utilization"][0]
                        
                        # Aggregate by account
                        agg_df = plot_df.groupby(acc_col)[[lim_col, util_col]].sum().reset_index()
                        
                        # Sort by Utilization to get top accounts
                        top_accounts = agg_df.nlargest(10, util_col)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=top_accounts[acc_col],
                            y=top_accounts[lim_col],
                            name='Facility Limit',
                            marker_color='#1f77b4'
                        ))
                        fig.add_trace(go.Bar(
                            x=top_accounts[acc_col],
                            y=top_accounts[util_col],
                            name='Utilization',
                            marker_color='#ff7f0e'
                        ))
                        
                        fig.update_layout(
                            barmode='group',
                            xaxis_tickangle=-45,
                            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Limit and Utilization data not found for bar chart.")
                
                # Chart 2: Risk / Status Distribution
                with c2:
                    if "Status" in mapping:
                        st.subheader("Account Status")
                        stat_col = mapping["Status"][0]
                        if "Account" in mapping:
                            stat_df = plot_df.groupby(stat_col)[acc_col].nunique().reset_index()
                            fig1 = px.pie(stat_df, names=stat_col, values=acc_col, hole=0.4)
                        else:
                            fig1 = px.pie(plot_df, names=stat_col, hole=0.4)
                            
                        # Improve donut chart layout
                        fig1.update_traces(textposition='inside', textinfo='percent+label')
                        fig1.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig1, use_container_width=True)
                    elif "DPD" in mapping:
                        st.subheader("Overdue Status")
                        dpd_col = mapping["DPD"][0]
                        # Create buckets
                        plot_df['DPD_Bucket'] = pd.cut(
                            pd.to_numeric(plot_df[dpd_col], errors='coerce'), 
                            bins=[-float('inf'), 0, 15, 30, float('inf')], 
                            labels=['Current', '1-15 Days', '16-30 Days', '30+ Days']
                        )
                        dpd_df = plot_df['DPD_Bucket'].value_counts().reset_index()
                        dpd_df.columns = ['Status', 'Count']
                        fig1 = px.pie(dpd_df, names='Status', values='Count', hole=0.4)
                        fig1.update_traces(textposition='inside', textinfo='percent+label')
                        fig1.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig1, use_container_width=True)
                    else:
                        st.info("Status or DPD data not found for pie chart.")

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
