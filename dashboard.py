import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io

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

# Custom CSS for minor styling (optional but clean)
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
@st.cache_data
def load_and_merge_data(single_file, master_file, view1_file, view2_file):
    """Parses and merges uploaded Excel/CSV files based on the input mode."""
    try:
        if single_file is not None:
            df = parse_file(single_file)
        elif master_file and view1_file and view2_file:
            df_master = parse_file(master_file)
            df_v1 = parse_file(view1_file)
            df_v2 = parse_file(view2_file)
            
            # Use Buyer as primary join key, fallback to Company if needed. 
            # In Pandas, doing a clean merge is best.
            # Assuming Buyer is the unique identifier across sheets.
            df = pd.merge(df_master, df_v1, on='Buyer', how='outer', suffixes=('', '_drop1'))
            df = pd.merge(df, df_v2, on='Buyer', how='outer', suffixes=('', '_drop2'))
            
            # Clean up duplicate columns from merges
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
    """Cleans data and calculates derived columns."""
    # Ensure all required columns exist, fill missing with NaN
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
            
    # Filter valid statuses
    df = df[df['Account_Status'].isin(VALID_STATUSES)].copy()
    
    # Coerce numerics
    df['Outstanding Balance'] = pd.to_numeric(df['Outstanding Balance'], errors='coerce').fillna(0)
    df['Facility Size'] = pd.to_numeric(df['Facility Size'], errors='coerce').fillna(0)
    df['Payment Total USD'] = pd.to_numeric(df['Payment Total USD'], errors='coerce').fillna(0)
    
    # Coerce dates
    df['Last Disbursed Date'] = pd.to_datetime(df['Last Disbursed Date'], errors='coerce')
    df['Due Date of Invoice'] = pd.to_datetime(df['Due Date of Invoice'], errors='coerce')
    df['Settlement Date'] = pd.to_datetime(df['Settlement Date'], errors='coerce')
    
    # AM Names logic
    df['AM Names'] = df['AM Names'].combine_first(df['AM'])
    df['AM Names'] = df['AM Names'].fillna('Unassigned')
    df.loc[df['Buyer'].isna() | (df['Buyer'] == ''), 'AM Names'] = 'Unassigned'
    
    # Utilisation %
    df['Utilisation %'] = (df['Outstanding Balance'] / df['Facility Size'].replace(0, pd.NA)) * 100
    df['Utilisation %'] = df['Utilisation %'].fillna(0)
    
    # 180-Day Alert
    now = pd.Timestamp.now()
    df['Days Since Disbursed'] = (now - df['Last Disbursed Date']).dt.days
    df['180-Day Alert'] = df['Days Since Disbursed'].apply(lambda x: 'YES' if pd.notna(x) and x > 180 else 'NO')
    
    # Present Month Rule
    curr_month = now.month
    curr_year = now.year
    df['In Current Month'] = False
    
    mask_due = (df['Due Date of Invoice'].dt.month == curr_month) & (df['Due Date of Invoice'].dt.year == curr_year)
    mask_settle = (df['Settlement Date'].dt.month == curr_month) & (df['Settlement Date'].dt.year == curr_year)
    df.loc[mask_due | mask_settle, 'In Current Month'] = True

    # Deduplicated Repayment
    # Group by Buyer and Settlement Date to sum Payment Total USD
    # To assign back to df without duplicating, we transform
    df['Buyer_Clean'] = df['Buyer'].fillna('Unknown')
    df['Settle_Date_Clean'] = df['Settlement Date'].dt.date.fillna('None')
    
    # Calculate dedup sum
    dedup_sum = df[df['Payment Total USD'] > 0].groupby(['Buyer_Clean', 'Settle_Date_Clean'])['Payment Total USD'].sum().reset_index()
    dedup_sum.rename(columns={'Payment Total USD': 'Deduplicated Repayment'}, inplace=True)
    
    df = pd.merge(df, dedup_sum, on=['Buyer_Clean', 'Settle_Date_Clean'], how='left')
    df['Deduplicated Repayment'] = df['Deduplicated Repayment'].fillna(0)
    
    return df

# --- UI LAYOUT ---

st.title("📊 AM Portfolio Dashboard")

# Sidebar - Data Upload
with st.sidebar:
    st.header("Data Upload")
    upload_mode = st.radio("Upload Mode", ["Single Combined Sheet", "Three Separate Sheets"])
    
    single_file = None
    master_file = None
    view1_file = None
    view2_file = None
    
    if upload_mode == "Single Combined Sheet":
        single_file = st.file_uploader("Upload Combined Data (Excel/CSV)", type=["xlsx", "csv"])
    else:
        master_file = st.file_uploader("1. Masterdata", type=["xlsx", "csv"])
        view1_file = st.file_uploader("2. View 1", type=["xlsx", "csv"])
        view2_file = st.file_uploader("3. View 2", type=["xlsx", "csv"])
        
    process_btn = st.button("Process Data", type="primary", use_container_width=True)

# Process logic
if 'master_df' not in st.session_state:
    st.session_state.master_df = None

if process_btn:
    with st.spinner("Processing data..."):
        df = load_and_merge_data(single_file, master_file, view1_file, view2_file)
        if df is not None:
            st.session_state.master_df = df
            st.success("Data successfully processed!")

if st.session_state.master_df is not None:
    df = st.session_state.master_df
    
    # Global Filters (Top Bar)
    st.markdown("### Global Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        am_options = ["All AMs"] + sorted(list(df['AM Names'].dropna().unique()))
        selected_am = st.selectbox("AM Name", am_options)
        
    with col2:
        status_options = ["All Statuses"] + sorted(list(df['Account_Status'].dropna().unique()))
        selected_status = st.selectbox("Account Status", status_options)
        
    with col3:
        search_query = st.text_input("Search Company or Buyer", placeholder="Type to search...")
        
    # Apply Global Filters
    filtered_df = df.copy()
    if selected_am != "All AMs":
        filtered_df = filtered_df[filtered_df['AM Names'] == selected_am]
    if selected_status != "All Statuses":
        filtered_df = filtered_df[filtered_df['Account_Status'] == selected_status]
    if search_query:
        query = search_query.lower()
        mask = filtered_df['Company'].str.lower().str.contains(query, na=False) | \
               filtered_df['Buyer'].str.lower().str.contains(query, na=False)
        filtered_df = filtered_df[mask]
        
    st.divider()
    
    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "💵 Invoices & Repayments", 
        "💤 Workable Inactive", 
        "🏆 Top 15 Accounts", 
        "👑 Leadership Direct"
    ])
    
    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.subheader("Overview / Masterdata Handover")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(
                filtered_df[['Company', 'Buyer', 'Account_Status', 'AM Names', 'Outstanding Balance', 'Facility Size', 'Utilisation %', '180-Day Alert']],
                use_container_width=True,
                hide_index=True
            )
            
        with c2:
            st.markdown("#### Utilisation Distribution")
            # Categorize
            def cat_util(u):
                if u >= 70: return 'High (≥70%)'
                elif u >= 50: return 'Medium (50-69%)'
                else: return 'Low (<50%)'
            
            util_series = filtered_df['Utilisation %'].apply(cat_util)
            counts = util_series.value_counts().reset_index()
            counts.columns = ['Category', 'Count']
            
            fig_util = px.pie(counts, names='Category', values='Count', hole=0.5,
                              color='Category',
                              color_discrete_map={'High (≥70%)':'#e53e3e', 'Medium (50-69%)':'#dd6b20', 'Low (<50%)':'#38a169'})
            st.plotly_chart(fig_util, use_container_width=True)

    # --- TAB 2: INVOICES & REPAYMENTS ---
    with tab2:
        st.subheader("Invoices & Repayments (Current Month)")
        
        month_df = filtered_df[filtered_df['In Current Month'] == True].copy()
        
        # Calculate Total Repayments (deduplicated)
        # We drop duplicate Buyer+SettleDate to sum the deduplicated amounts
        unique_repayments = month_df.drop_duplicates(subset=['Buyer_Clean', 'Settle_Date_Clean'])
        total_repayments = unique_repayments['Deduplicated Repayment'].sum()
        
        st.metric("Total Repayments (Current Month)", f"${total_repayments:,.2f}")
        
        # Determine Flags
        def get_invoice_flags(row):
            flags = []
            if pd.isna(row['Settlement Date']):
                flags.append("Collect Amount")
            
            # Low Utilisation Post-Repayment
            if row['Facility Size'] > 0:
                post_util = (row['Outstanding Balance'] - row['Deduplicated Repayment']) / row['Facility Size']
                if post_util < 0.5:
                    flags.append("Low Util Post-Repay")
            return ", ".join(flags)
            
        month_df['Flags'] = month_df.apply(get_invoice_flags, axis=1)
        
        # Avoid showing the same deduplicated repayment row multiple times unnecessarily,
        # but since we might have multiple due dates, we show unique rows.
        # Clean display cols
        display_month = month_df[['Buyer', 'Company', 'AM Names', 'Due Date of Invoice', 'Settlement Date', 'Deduplicated Repayment', 'Flags']]
        display_month = display_month.drop_duplicates()
        
        st.dataframe(display_month, use_container_width=True, hide_index=True)

    # --- TAB 3: WORKABLE INACTIVE AM ---
    with tab3:
        st.subheader("Workable Inactive AM")
        
        inactive_df = filtered_df[filtered_df['Account_Status'] == 'Workable-Inactive (AM)'].copy()
        
        i1, i2, i3 = st.columns(3)
        i1.metric("Inactive Accounts", len(inactive_df))
        i2.metric("Total Outstanding Balance", f"${inactive_df['Outstanding Balance'].sum():,.2f}")
        i3.metric("Total Facility Size", f"${inactive_df['Facility Size'].sum():,.2f}")
        
        st.dataframe(
            inactive_df[['Buyer', 'Company', 'AM Names', 'Outstanding Balance', 'Facility Size']],
            use_container_width=True, hide_index=True
        )

    # --- TAB 4: TOP 15 ACCOUNTS ---
    with tab4:
        st.subheader("Top 15 Accounts (by Facility Size)")
        
        top15 = filtered_df.sort_values(by='Facility Size', ascending=False).head(15).copy()
        top15['Rank'] = range(1, len(top15) + 1)
        
        t1, t2 = st.columns([1, 1])
        with t1:
            st.dataframe(
                top15[['Rank', 'Buyer', 'Company', 'Facility Size', 'Outstanding Balance', 'Utilisation %', 'AM Names']],
                use_container_width=True, hide_index=True
            )
            
        with t2:
            if not top15.empty:
                # Melt for grouped bar chart
                melted = top15.melt(id_vars=['Buyer'], value_vars=['Facility Size', 'Outstanding Balance'], var_name='Metric', value_name='Amount')
                fig_top15 = px.bar(melted, x='Amount', y='Buyer', color='Metric', orientation='h', barmode='group')
                fig_top15.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_top15, use_container_width=True)

    # --- TAB 5: LEADERSHIP DIRECT ---
    with tab5:
        st.subheader("Leadership Direct (Top 50)")
        
        direct_df = filtered_df[filtered_df['Account_Status'] == 'Team Direct'].copy()
        top50 = direct_df.sort_values(by='Facility Size', ascending=False).head(50).copy()
        top50['Rank'] = range(1, len(top50) + 1)
        
        l1, l2 = st.columns([1, 1])
        with l1:
            st.dataframe(
                top50[['Rank', 'Buyer', 'Company', 'Facility Size', 'Utilisation %']],
                use_container_width=True, hide_index=True
            )
            
        with l2:
            if not top50.empty:
                fig_lead = px.pie(top50.head(10), names='Buyer', values='Facility Size', hole=0.4, title="Top 10 Direct Distribution")
                st.plotly_chart(fig_lead, use_container_width=True)
else:
    st.info("👈 Please upload your data files in the sidebar and click 'Process Data'.")