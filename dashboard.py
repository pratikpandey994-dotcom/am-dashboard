import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
def generate_sample_data():
    """Generates a realistic dummy dataset for immediate testing."""
    np.random.seed(42)
    num_rows = 150
    now = pd.Timestamp.now()
    
    buyers = [f"Buyer Corp {i}" for i in range(1, 51)]
    companies = [f"Supplier Inc {i}" for i in range(1, 21)]
    ams = ["Alice Johnson", "Bob Smith", "Charlie Davis", "Diana Prince", "Unassigned"]
    
    data = []
    for _ in range(num_rows):
        buyer = np.random.choice(buyers)
        am = np.random.choice(ams)
        fac_size = np.random.uniform(50000, 2000000)
        out_bal = fac_size * np.random.uniform(0.1, 1.1)
        
        if np.random.random() < 0.2:
            last_disb = now - timedelta(days=np.random.randint(181, 300))
        else:
            last_disb = now - timedelta(days=np.random.randint(5, 170))
            
        if np.random.random() < 0.3:
            due_date = now + timedelta(days=np.random.randint(-15, 15))
        else:
            due_date = now - timedelta(days=np.random.randint(30, 365 * 3)) # Historical data
            
        settle_date = due_date if np.random.random() < 0.5 else pd.NaT
        repay = out_bal * np.random.uniform(0.1, 0.5) if pd.notna(settle_date) else 0
        
        data.append({
            'Buyer': buyer,
            'Company': np.random.choice(companies),
            'Account_Status': np.random.choice(VALID_STATUSES),
            'AM Names': am,
            'AM': am,
            'Facility Size': fac_size,
            'Outstanding Balance': out_bal,
            'Last Disbursed Date': last_disb,
            'Due Date of Invoice': due_date,
            'Settlement Date': settle_date,
            'Payment Total USD': repay
        })
        
    return pd.DataFrame(data)

@st.cache_data
def load_and_merge_data(single_file, master_file, view1_file, view2_file):
    try:
        if single_file is not None:
            df = parse_file(single_file)
            return process_data(df)
        elif master_file and view1_file and view2_file:
            df_master = parse_file(master_file)
            df_v1 = parse_file(view1_file)
            df_v2 = parse_file(view2_file)
            
            # --- MAPPING LOGIC ---
            if 'Facility_Size' in df_master.columns: df_master = df_master.rename(columns={'Facility_Size': 'Facility Size'})
            if 'OB' in df_master.columns: df_master = df_master.rename(columns={'OB': 'Master_OB'})
            if 'AM' in df_master.columns: df_master = df_master.rename(columns={'AM': 'Master_AM'})
            
            # Clean Account Status rigorously
            if 'Account_Status' in df_master.columns:
                # Remove spaces around hyphens and title case it so "temporarily suspended" becomes "Temporarily Suspended"
                df_master['Account_Status'] = df_master['Account_Status'].astype(str).str.replace(r'\s*-\s*', '-', regex=True).str.strip().str.title()
                # Fix specific casing for (AM)
                df_master['Account_Status'] = df_master['Account_Status'].str.replace('(Am)', '(AM)', regex=False)
            
            if 'company' in df_v1.columns: df_v1 = df_v1.rename(columns={'company': 'Buyer'})
            if 'Outstanding_Balance' in df_v1.columns: df_v1 = df_v1.rename(columns={'Outstanding_Balance': 'View1_OB'})
            if 'Last_Disbursed_Date' in df_v1.columns: df_v1 = df_v1.rename(columns={'Last_Disbursed_Date': 'Last Disbursed Date'})
            if 'AM_Name' in df_v1.columns: df_v1 = df_v1.rename(columns={'AM_Name': 'View1_AM'})
                
            if 'due_date_of_invoice' in df_v2.columns: df_v2 = df_v2.rename(columns={'due_date_of_invoice': 'Due Date of Invoice'})
            if 'settlement_date' in df_v2.columns: df_v2 = df_v2.rename(columns={'settlement_date': 'Settlement Date'})
            if 'payment_total_usd' in df_v2.columns: df_v2 = df_v2.rename(columns={'payment_total_usd': 'Payment Total USD'})
            
            # Merge Master + View 1
            df = pd.merge(df_master, df_v1, on='Buyer', how='outer', suffixes=('', '_drop1'))
            
            # Resolve combined columns (Prioritize View 1 for OB, Master for AM)
            df['Outstanding Balance'] = df.get('View1_OB', pd.Series(dtype=float)).combine_first(df.get('Master_OB', pd.Series(dtype=float)))
            df['AM Names'] = df.get('Master_AM', pd.Series(dtype=str)).combine_first(df.get('View1_AM', pd.Series(dtype=str)))
            df['Company'] = df['Buyer'] 
            
            # Master/V1 + View 2
            df = pd.merge(df, df_v2, on='Buyer', how='outer', suffixes=('', '_drop2'))
            
            df = df.loc[:, ~df.columns.str.endswith('_drop1')]
            df = df.loc[:, ~df.columns.str.endswith('_drop2')]
            
            return process_data(df)
        else:
            return None
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
            
    # Apply Strict Filter for Valid Statuses
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

    df['Buyer_Clean'] = df['Buyer'].fillna('Unknown')
    df['Settle_Date_Clean'] = df['Settlement Date'].dt.date.fillna('None')
    
    # Pre-calculate Dedup Logic globally
    dedup_sum = df[df['Payment Total USD'] > 0].groupby(['Buyer_Clean', 'Settle_Date_Clean'])['Payment Total USD'].sum().reset_index()
    dedup_sum.rename(columns={'Payment Total USD': 'Deduplicated Repayment'}, inplace=True)
    df = pd.merge(df, dedup_sum, on=['Buyer_Clean', 'Settle_Date_Clean'], how='left')
    df['Deduplicated Repayment'] = df['Deduplicated Repayment'].fillna(0)
    
    return df

# --- UI LAYOUT ---
st.title("📊 Unified AM Portfolio Dashboard")

if 'master_df' not in st.session_state:
    st.session_state.master_df = None

with st.sidebar:
    st.header("📂 Data Source")
    
    if st.button("Load Sample Data", type="primary", use_container_width=True):
        raw_sample = generate_sample_data()
        st.session_state.master_df = process_data(raw_sample)
        st.success("Sample data loaded!")
        st.rerun()
        
    st.divider()
    
    upload_mode = st.radio("Upload Own Files:", ["Single Combined Sheet", "Three Separate Sheets"])
    
    single_file = master_file = view1_file = view2_file = None
    if upload_mode == "Single Combined Sheet":
        single_file = st.file_uploader("Upload Combined Data (Excel/CSV)", type=["xlsx", "csv"])
    else:
        master_file = st.file_uploader("1. Masterdata", type=["xlsx", "csv"])
        view1_file = st.file_uploader("2. View 1", type=["xlsx", "csv"])
        view2_file = st.file_uploader("3. View 2", type=["xlsx", "csv"])
        
    if st.button("Process Files", use_container_width=True):
        with st.spinner("Processing logic & Mapping Columns..."):
            df = load_and_merge_data(single_file, master_file, view1_file, view2_file)
            if df is not None:
                st.session_state.master_df = df
                st.success("Data successfully processed & mapped!")
                st.rerun()

if st.session_state.master_df is not None:
    df = st.session_state.master_df
    
    # 1. Available Months Logic (Fixing the Empty Month Bug)
    all_dates = pd.concat([df['Due Date of Invoice'], df['Settlement Date']]).dropna()
    if not all_dates.empty:
        available_months = sorted(all_dates.dt.to_period('M').unique().astype(str), reverse=True)
    else:
        available_months = [pd.Timestamp.now().strftime('%Y-%m')]

    # 2. Global Filters
    st.markdown("### 🎛️ Global Portfolio Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_month = st.selectbox("Selected Month (Invoices)", available_months)
    with col2:
        am_options = ["All AMs"] + sorted(list(df['AM Names'].dropna().unique()))
        selected_am = st.selectbox("AM Name", am_options)
    with col3:
        status_options = ["All Statuses"] + sorted(list(df['Account_Status'].dropna().unique()))
        selected_status = st.selectbox("Account Status", status_options)
    with col4:
        search_query = st.text_input("Search Company or Buyer", placeholder="Type to search...")
        
    # Apply Global Filters
    filtered_df = df.copy()
    if selected_am != "All AMs": filtered_df = filtered_df[filtered_df['AM Names'] == selected_am]
    if selected_status != "All Statuses": filtered_df = filtered_df[filtered_df['Account_Status'] == selected_status]
    if search_query:
        query = search_query.lower()
        mask = filtered_df['Company'].str.lower().str.contains(query, na=False) | filtered_df['Buyer'].str.lower().str.contains(query, na=False)
        filtered_df = filtered_df[mask]
        
    # Apply Month Filter for the specific 'month_df'
    filtered_df['In Selected Month'] = False
    mask_due = (filtered_df['Due Date of Invoice'].dt.to_period('M').astype(str) == selected_month)
    mask_settle = (filtered_df['Settlement Date'].dt.to_period('M').astype(str) == selected_month)
    filtered_df.loc[mask_due | mask_settle, 'In Selected Month'] = True
    
    month_df = filtered_df[filtered_df['In Selected Month'] == True].copy()
    
    # 3. Separate the Unique Accounts (Fixing the Duplicate Multiplication Bug)
    # We take exactly ONE row per buyer for facility/outstanding KPI math so we don't multiply by # of invoices
    unique_accts = filtered_df.drop_duplicates(subset=['Buyer_Clean']).copy()
    
    st.divider()

    # --- TOP ROW KPIs ---
    tot_fac = unique_accts['Facility Size'].sum()
    tot_out = unique_accts['Outstanding Balance'].sum()
    glob_util = (tot_out / tot_fac * 100) if tot_fac > 0 else 0
    
    # Repayments KPI (Deduplicated properly)
    unique_repayments = month_df.drop_duplicates(subset=['Buyer_Clean', 'Settle_Date_Clean'])
    tot_repay = unique_repayments['Deduplicated Repayment'].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Facility Size", f"${tot_fac:,.0f}")
    k2.metric("Total Outstanding", f"${tot_out:,.0f}")
    k3.metric("Global Average Utilisation", f"{glob_util:.1f}%")
    k4.metric(f"Collections ({selected_month})", f"${tot_repay:,.0f}")
    
    st.divider()

    # --- MIDDLE ROW 1: PORTFOLIO & RISK ---
    c1, c2 = st.columns([6, 4])
    with c1:
        st.markdown("#### 🌍 Portfolio Concentration")
        fig_tree = px.treemap(
            unique_accts, 
            path=[px.Constant("All Portfolio"), 'Account_Status', 'AM Names', 'Buyer_Clean'], 
            values='Facility Size',
            color='Utilisation %',
            color_continuous_scale='RdYlGn_r',
            range_color=[0, 100],
            title="Sized by Facility, Colored by Utilisation"
        )
        fig_tree.update_traces(root_color="lightgrey")
        fig_tree.update_layout(margin = dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_tree, use_container_width=True)
        
    with c2:
        st.markdown("#### ⚠️ Risk Matrix (Facility vs Outstanding)")
        max_val = max(unique_accts['Facility Size'].max(), unique_accts['Outstanding Balance'].max()) if not unique_accts.empty else 100
        fig_scatter = px.scatter(
            unique_accts, x="Facility Size", y="Outstanding Balance", 
            color="180-Day Alert", 
            color_discrete_map={"YES": "red", "NO": "green"},
            hover_name="Buyer_Clean", hover_data=["AM Names", "Utilisation %"],
            title="Red dots indicate 180-Day Disbursement Alerts"
        )
        if max_val > 0:
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="black", dash="dash"))
        fig_scatter.update_layout(margin = dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # --- MIDDLE ROW 2: TRENDS & LEADERBOARD ---
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"#### 💵 Daily Collections Trend ({selected_month})")
        if not unique_repayments.empty:
            timeline_df = unique_repayments.groupby('Settle_Date_Clean')['Deduplicated Repayment'].sum().reset_index()
            timeline_df = timeline_df[timeline_df['Settle_Date_Clean'] != 'None']
            fig_timeline = px.bar(
                timeline_df, x='Settle_Date_Clean', y='Deduplicated Repayment',
                labels={'Settle_Date_Clean': 'Settlement Date', 'Deduplicated Repayment': 'Amount Collected'}
            )
            fig_timeline.update_layout(margin = dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info(f"No settlements found for {selected_month}.")

    with c4:
        st.markdown("#### 👑 AM Performance Leaderboard")
        am_perf = unique_accts.groupby('AM Names').agg(
            Total_Facility=('Facility Size', 'sum'),
            Avg_Utilisation=('Utilisation %', 'mean')
        ).reset_index().sort_values('Total_Facility', ascending=False)
        
        fig_combo = go.Figure()
        fig_combo.add_trace(go.Bar(x=am_perf['AM Names'], y=am_perf['Total_Facility'], name="Facility Size ($)", marker_color='teal', yaxis='y1'))
        fig_combo.add_trace(go.Scatter(x=am_perf['AM Names'], y=am_perf['Avg_Utilisation'], name="Avg Utilisation (%)", mode='lines+markers', marker=dict(color='orange', size=8), line=dict(width=3), yaxis='y2'))
        fig_combo.update_layout(
            yaxis=dict(title="Facility Size ($)", side='left'),
            yaxis2=dict(title="Avg Utilisation (%)", overlaying='y', side='right', range=[0, 100]),
            legend=dict(x=1.1, y=1),
            margin = dict(t=10, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_combo, use_container_width=True)

    st.divider()

    # --- MASTER DATA TABLE ---
    st.markdown(f"### 📋 Unified Master Data ({selected_month})")
    st.write("All flags and logics (180-Day, Present Month, Low Utilisation) are pre-calculated and merged into this unified table.")
    
    # Calculate Post-Repayment flags specifically for the selected month
    def get_global_flags(row):
        flags = []
        if pd.isna(row['Settlement Date']): 
            flags.append("Collect Amount (Missing Settle Date)")
        if row['Facility Size'] > 0:
            post_util = (row['Outstanding Balance'] - row['Deduplicated Repayment']) / row['Facility Size']
            if post_util < 0.5: flags.append("Low Util Post-Repay")
        return " | ".join(flags) if flags else "None"
        
    month_df['Invoice Flags'] = month_df.apply(get_global_flags, axis=1)
    
    display_cols = [
        'Company', 'Buyer', 'Account_Status', 'AM Names', 
        'Outstanding Balance', 'Facility Size', 'Utilisation %', 'Utilisation Category',
        '180-Day Alert', 'Due Date of Invoice', 'Settlement Date', 
        'Deduplicated Repayment', 'Invoice Flags'
    ]
    
    table_df = month_df[display_cols].copy()
    for date_col in ['Due Date of Invoice', 'Settlement Date']:
        table_df[date_col] = table_df[date_col].dt.strftime('%Y-%m-%d').fillna('')
        
    st.dataframe(table_df, use_container_width=True, hide_index=True)

else:
    st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h1 style="color: #008080;">Welcome to AM Portfolio Intelligence</h1>
            <p style="color: #666; font-size: 18px;">To see the dashboard in action, please <b>Load Sample Data</b> or <b>Upload your 3 Excel files</b> using the sidebar on the left.</p>
        </div>
    """, unsafe_allow_html=True)