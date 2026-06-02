import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(page_title="AM Portfolio Dashboard", layout="wide", page_icon="📊")

# --- CONSTANTS & CONFIG ---
VALID_STATUSES = [
    'Workable-Active',
    'Workable-Inactive (AM)',
    'Workable-Temporarily Suspended',
    'Team Direct'
]

REQUIRED_COLS = [
    'Buyer', 'Account_Status', 'AM Names', 'Company', 'Outstanding Balance',
    'Facility Size', 'Due Date of Invoice', 'Settlement Date', 'AM',
    'Payment Total USD', 'Last Disbursed Date'
]

# --- UTILITY FUNCTIONS ---
@st.cache_data
def load_and_merge_data(single_file, master_file, view1_file, view2_file):
    try:
        if single_file is not None:
            df = parse_file(single_file)
        elif master_file and view1_file and view2_file:
            df_master = parse_file(master_file)
            df_v1 = parse_file(view1_file)
            df_v2 = parse_file(view2_file)
            
            df = pd.merge(df_master, df_v1, on='Buyer', how='outer', suffixes=('', '_drop1'))
            df = pd.merge(df, df_v2, on='Buyer', how='outer', suffixes=('', '_drop2'))
            
            df = df.loc[:, ~df.columns.str.endswith('_drop1')]
            df = df.loc[:, ~df.columns.str.endswith('_drop2')]
        else:
            return None
        
        return process_data(df)
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None

def parse_file(file):
    if file.name.lower().endswith('.csv'):
        return pd.read_csv(file)
    return pd.read_excel(file)

def process_data(df):
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
            
    df = df[df['Account_Status'].isin(VALID_STATUSES)].copy()
    
    df['Outstanding Balance'] = pd.to_numeric(df['Outstanding Balance'], errors='coerce').fillna(0)
    df['Facility Size'] = pd.to_numeric(df['Facility Size'], errors='coerce').fillna(0)
    df['Payment Total USD'] = pd.to_numeric(df['Payment Total USD'], errors='coerce').fillna(0)
    
    df['Last Disbursed Date'] = pd.to_datetime(df['Last Disbursed Date'], errors='coerce')
    df['Due Date of Invoice'] = pd.to_datetime(df['Due Date of Invoice'], errors='coerce')
    df['Settlement Date'] = pd.to_datetime(df['Settlement Date'], errors='coerce')
    
    df['AM Names'] = df['AM Names'].combine_first(df['AM'])
    df['AM Names'] = df['AM Names'].fillna('Unassigned')
    df.loc[df['Buyer'].isna() | (df['Buyer'] == ''), 'AM Names'] = 'Unassigned'
    
    df['Utilisation %'] = (df['Outstanding Balance'] / df['Facility Size'].replace(0, pd.NA)) * 100
    df['Utilisation %'] = df['Utilisation %'].fillna(0)
    
    def cat_util(u):
        if u >= 70: return 'High'
        elif u >= 50: return 'Medium'
        else: return 'Low'
    df['Utilisation Category'] = df['Utilisation %'].apply(cat_util)
    
    now = pd.Timestamp.now()
    df['Days Since Disbursed'] = (now - df['Last Disbursed Date']).dt.days
    df['180-Day Alert'] = df['Days Since Disbursed'].apply(lambda x: 'YES' if pd.notna(x) and x > 180 else 'NO')
    
    curr_month = now.month
    curr_year = now.year
    df['In Current Month'] = False
    mask_due = (df['Due Date of Invoice'].dt.month == curr_month) & (df['Due Date of Invoice'].dt.year == curr_year)
    mask_settle = (df['Settlement Date'].dt.month == curr_month) & (df['Settlement Date'].dt.year == curr_year)
    df.loc[mask_due | mask_settle, 'In Current Month'] = True

    df['Buyer_Clean'] = df['Buyer'].fillna('Unknown')
    df['Settle_Date_Clean'] = df['Settlement Date'].dt.date.fillna('None')
    
    dedup_sum = df[df['Payment Total USD'] > 0].groupby(['Buyer_Clean', 'Settle_Date_Clean'])['Payment Total USD'].sum().reset_index()
    dedup_sum.rename(columns={'Payment Total USD': 'Deduplicated Repayment'}, inplace=True)
    
    df = pd.merge(df, dedup_sum, on=['Buyer_Clean', 'Settle_Date_Clean'], how='left')
    df['Deduplicated Repayment'] = df['Deduplicated Repayment'].fillna(0)
    
    return df

# --- UI LAYOUT ---
st.title("📊 AM Portfolio Intelligence")
st.markdown("Advanced visualization and risk analysis based on core portfolio logic.")

# Sidebar - Data Upload
with st.sidebar:
    st.header("Data Upload")
    upload_mode = st.radio("Upload Mode", ["Single Combined Sheet", "Three Separate Sheets"])
    
    single_file = master_file = view1_file = view2_file = None
    
    if upload_mode == "Single Combined Sheet":
        single_file = st.file_uploader("Upload Combined Data (Excel/CSV)", type=["xlsx", "csv"])
    else:
        master_file = st.file_uploader("1. Masterdata", type=["xlsx", "csv"])
        view1_file = st.file_uploader("2. View 1", type=["xlsx", "csv"])
        view2_file = st.file_uploader("3. View 2", type=["xlsx", "csv"])
        
    process_btn = st.button("Process Data", type="primary", use_container_width=True)

if 'master_df' not in st.session_state:
    st.session_state.master_df = None

if process_btn:
    with st.spinner("Processing logic..."):
        df = load_and_merge_data(single_file, master_file, view1_file, view2_file)
        if df is not None:
            st.session_state.master_df = df
            st.success("Data successfully processed!")

if st.session_state.master_df is not None:
    df = st.session_state.master_df
    
    # Global Filters
    st.markdown("### Global Portfolio Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        am_options = ["All AMs"] + sorted(list(df['AM Names'].dropna().unique()))
        selected_am = st.selectbox("AM Name", am_options)
    with col2:
        status_options = ["All Statuses"] + sorted(list(df['Account_Status'].dropna().unique()))
        selected_status = st.selectbox("Account Status", status_options)
    with col3:
        search_query = st.text_input("Search Company or Buyer", placeholder="Type to search...")
        
    filtered_df = df.copy()
    if selected_am != "All AMs": filtered_df = filtered_df[filtered_df['AM Names'] == selected_am]
    if selected_status != "All Statuses": filtered_df = filtered_df[filtered_df['Account_Status'] == selected_status]
    if search_query:
        query = search_query.lower()
        mask = filtered_df['Company'].str.lower().str.contains(query, na=False) | filtered_df['Buyer'].str.lower().str.contains(query, na=False)
        filtered_df = filtered_df[mask]
        
    st.divider()
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Executive Overview", 
        "⚠️ Risk & Exposure", 
        "💵 Cashflow & Collections", 
        "👑 AM Performance"
    ])
    
    # --- TAB 1: EXECUTIVE OVERVIEW ---
    with tab1:
        st.subheader("Global Portfolio Structure")
        
        # Unique accounts for treemap to avoid massive duplication
        unique_accts = filtered_df.drop_duplicates(subset=['Buyer_Clean']).copy()
        
        tot_fac = unique_accts['Facility Size'].sum()
        tot_out = unique_accts['Outstanding Balance'].sum()
        glob_util = (tot_out / tot_fac * 100) if tot_fac > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Facility Size", f"${tot_fac:,.0f}")
        k2.metric("Total Outstanding", f"${tot_out:,.0f}")
        k3.metric("Global Average Utilisation", f"{glob_util:.1f}%")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Portfolio Concentration (Status ➔ AM ➔ Buyer)")
            # Treemap
            fig_tree = px.treemap(
                unique_accts, 
                path=[px.Constant("All Portfolio"), 'Account_Status', 'AM Names', 'Buyer_Clean'], 
                values='Facility Size',
                color='Utilisation %',
                color_continuous_scale='RdYlGn_r', # Red is high utilization, Green is low
                range_color=[0, 100]
            )
            fig_tree.update_traces(root_color="lightgrey")
            fig_tree.update_layout(margin = dict(t=20, l=20, r=20, b=20))
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with c2:
            st.markdown("#### Global Utilisation Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = glob_util,
                title = {'text': "Utilisation %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 70], 'color': "gold"},
                        {'range': [70, 100], 'color': "salmon"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with st.expander("View Raw Masterdata"):
            st.dataframe(unique_accts[['Buyer', 'Company', 'Account_Status', 'AM Names', 'Facility Size', 'Outstanding Balance', 'Utilisation %']], hide_index=True)

    # --- TAB 2: RISK & EXPOSURE ---
    with tab2:
        st.subheader("Risk Identification & 180-Day Alerts")
        st.write("Accounts heavily utilized or flagged by the 180-Day disbursement rule.")
        
        unique_accts = filtered_df.drop_duplicates(subset=['Buyer_Clean']).copy()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Facility vs Outstanding (Risk Matrix)")
            # Scatter Plot
            # Create a line showing 100% utilisation (X=Y)
            max_val = max(unique_accts['Facility Size'].max(), unique_accts['Outstanding Balance'].max())
            fig_scatter = px.scatter(
                unique_accts, x="Facility Size", y="Outstanding Balance", 
                color="180-Day Alert", 
                color_discrete_map={"YES": "red", "NO": "green"},
                hover_name="Buyer_Clean", hover_data=["AM Names", "Utilisation %"]
            )
            # Add 100% utilisation reference line
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="black", dash="dash"))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with c2:
            st.markdown("#### Risk Profile by AM (Utilisation Categories)")
            # Stacked Bar
            risk_counts = unique_accts.groupby(['AM Names', 'Utilisation Category']).size().reset_index(name='Count')
            fig_bar = px.bar(
                risk_counts, x="AM Names", y="Count", color="Utilisation Category",
                color_discrete_map={"High": "#e53e3e", "Medium": "#dd6b20", "Low": "#38a169"},
                barmode='stack'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("View Accounts Requiring Attention (180-Day Alert or High Utilisation)"):
            risk_df = unique_accts[(unique_accts['180-Day Alert'] == 'YES') | (unique_accts['Utilisation Category'] == 'High')]
            st.dataframe(risk_df[['Buyer', 'AM Names', '180-Day Alert', 'Utilisation %', 'Outstanding Balance', 'Facility Size']], hide_index=True)

    # --- TAB 3: CASHFLOW & COLLECTIONS ---
    with tab3:
        st.subheader("Invoices & Deduplicated Repayments (Current Month)")
        
        # Apply strict present month rule
        month_df = filtered_df[filtered_df['In Current Month'] == True].copy()
        unique_repayments = month_df.drop_duplicates(subset=['Buyer_Clean', 'Settle_Date_Clean']).copy()
        
        tot_repay = unique_repayments['Deduplicated Repayment'].sum()
        st.metric("Total Deduplicated Repayments (Current Month)", f"${tot_repay:,.2f}")
        
        # Time-Series Bar Chart for Collections
        # Group by Settlement Date
        if not unique_repayments.empty:
            timeline_df = unique_repayments.groupby('Settle_Date_Clean')['Deduplicated Repayment'].sum().reset_index()
            # Filter out 'None' dates if any
            timeline_df = timeline_df[timeline_df['Settle_Date_Clean'] != 'None']
            
            fig_timeline = px.bar(
                timeline_df, x='Settle_Date_Clean', y='Deduplicated Repayment',
                title="Daily Collections Trend (Present Month)",
                labels={'Settle_Date_Clean': 'Settlement Date', 'Deduplicated Repayment': 'Amount Collected'}
            )
            fig_timeline.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
        with st.expander("View Invoice & Repayment Data"):
            def get_invoice_flags(row):
                flags = []
                if pd.isna(row['Settlement Date']): flags.append("Collect Amount")
                if row['Facility Size'] > 0:
                    post_util = (row['Outstanding Balance'] - row['Deduplicated Repayment']) / row['Facility Size']
                    if post_util < 0.5: flags.append("Low Util Post-Repay")
                return ", ".join(flags)
                
            month_df['Flags'] = month_df.apply(get_invoice_flags, axis=1)
            display_month = month_df[['Buyer', 'Company', 'AM Names', 'Due Date of Invoice', 'Settlement Date', 'Deduplicated Repayment', 'Flags']].drop_duplicates()
            st.dataframe(display_month, use_container_width=True, hide_index=True)

    # --- TAB 4: AM PERFORMANCE ---
    with tab4:
        st.subheader("Account Manager Performance Leaderboard")
        st.write("Comparing total portfolio size managed vs the average utilisation of those portfolios.")
        
        unique_accts = filtered_df.drop_duplicates(subset=['Buyer_Clean']).copy()
        
        am_perf = unique_accts.groupby('AM Names').agg(
            Total_Facility=('Facility Size', 'sum'),
            Avg_Utilisation=('Utilisation %', 'mean'),
            Account_Count=('Buyer_Clean', 'count')
        ).reset_index()
        
        am_perf = am_perf.sort_values('Total_Facility', ascending=False)
        
        # Combo Chart using Graph Objects
        fig_combo = go.Figure()
        
        # Bar chart for Facility Size
        fig_combo.add_trace(go.Bar(
            x=am_perf['AM Names'],
            y=am_perf['Total_Facility'],
            name="Total Facility Size ($)",
            marker_color='teal',
            yaxis='y1'
        ))
        
        # Line chart for Avg Utilisation
        fig_combo.add_trace(go.Scatter(
            x=am_perf['AM Names'],
            y=am_perf['Avg_Utilisation'],
            name="Avg Utilisation (%)",
            mode='lines+markers',
            marker=dict(color='orange', size=10),
            line=dict(width=3),
            yaxis='y2'
        ))
        
        # Layout for dual y-axis
        fig_combo.update_layout(
            title="Portfolio Size vs Utilisation Rate",
            yaxis=dict(title="Facility Size ($)", side='left'),
            yaxis2=dict(title="Avg Utilisation (%)", overlaying='y', side='right', range=[0, 100]),
            legend=dict(x=1.1, y=1)
        )
        st.plotly_chart(fig_combo, use_container_width=True)
        
        with st.expander("View AM Performance Data"):
            st.dataframe(am_perf, use_container_width=True, hide_index=True)

else:
    st.info("👈 Please upload your data files in the sidebar and click 'Process Data'.")