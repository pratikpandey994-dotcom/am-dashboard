import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="AM Portfolio Intelligence", layout="wide", page_icon="📊")

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 5px solid #00C4B4;
    }
    h1, h2, h3 {
        color: #F0F2F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA PROCESSING (FLEXIBLE INGESTION) ---

def apply_business_logic(df):
    """Applies the Phase 3 business logic to a unified dataframe and splits into Portfolio & V2"""
    # Ensure columns exist to prevent crashes on single sheets missing data
    required = ['buyer', 'account_status', 'am_names', 'facility_size', 'outstanding_balance', 
                'last_disbursed_date', 'due_date_invoice', 'settlement_date', 'payment_total_usd', 'am_view2']
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    # Clean Account Status
    df['account_status'] = df['account_status'].astype(str).str.replace(r'\s*-\s*', '-', regex=True).str.strip().str.title()
    df['account_status'] = df['account_status'].str.replace('(Am)', '(AM)', regex=False)
    valid_statuses = ['Workable-Active', 'Workable-Inactive (AM)', 'Workable-Temporarily Suspended', 'Team Direct']
    df = df[df['account_status'].isin(valid_statuses)].copy()

    df['am_names'] = df['am_names'].fillna('Unassigned')
    df.loc[df['am_names'] == '', 'am_names'] = 'Unassigned'
    
    df['facility_size'] = pd.to_numeric(df['facility_size'], errors='coerce').fillna(0)
    df['outstanding_balance'] = pd.to_numeric(df['outstanding_balance'], errors='coerce').fillna(0)
    
    now = pd.Timestamp.now()
    df['last_disbursed_date'] = pd.to_datetime(df['last_disbursed_date'], errors='coerce')
    df['due_date_invoice'] = pd.to_datetime(df['due_date_invoice'], errors='coerce')
    df['settlement_date'] = pd.to_datetime(df['settlement_date'], errors='coerce')
    df['payment_total_usd'] = pd.to_numeric(df['payment_total_usd'], errors='coerce').fillna(0)
    
    # Extract Portfolio (1 row per buyer)
    portfolio = df.drop_duplicates(subset=['buyer']).copy()
    
    portfolio['utilisation_pct'] = np.where(portfolio['facility_size'] > 0, (portfolio['outstanding_balance'] / portfolio['facility_size']) * 100, 0)
    portfolio['utilisation_category'] = pd.cut(portfolio['utilisation_pct'], bins=[-np.inf, 50, 70, np.inf], labels=['Low', 'Medium', 'High'], right=False)
    portfolio['alert_180_days'] = (now - portfolio['last_disbursed_date']).dt.days > 180

    # Extract V2 (All rows / Invoices)
    v2 = df.copy()
    v2['collect_amount_flag'] = v2['settlement_date'].isna()
    
    curr_month, curr_year = now.month, now.year
    mask_due = (v2['due_date_invoice'].dt.month == curr_month) & (v2['due_date_invoice'].dt.year == curr_year)
    mask_settle = (v2['settlement_date'].dt.month == curr_month) & (v2['settlement_date'].dt.year == curr_year)
    v2['present_month_rule'] = mask_due | mask_settle

    # Dedup Repayments
    v2['settlement_date_clean'] = v2['settlement_date'].dt.date.fillna('None')
    dedup = v2[v2['payment_total_usd'] > 0].groupby(['buyer', 'settlement_date_clean'])['payment_total_usd'].sum().reset_index()
    dedup = dedup.rename(columns={'payment_total_usd': 'deduped_repayment'})
    
    dedup_totals = dedup.groupby('buyer')['deduped_repayment'].sum().reset_index()
    portfolio = pd.merge(portfolio, dedup_totals, on='buyer', how='left')
    portfolio['deduped_repayment'] = portfolio['deduped_repayment'].fillna(0)
    
    portfolio['adjusted_outstanding'] = portfolio['outstanding_balance'] - portfolio['deduped_repayment']
    portfolio['post_repayment_util'] = np.where(portfolio['facility_size'] > 0, (portfolio['adjusted_outstanding'] / portfolio['facility_size']) * 100, 0)
    portfolio['low_util_post_repay_flag'] = portfolio['post_repayment_util'] < 50
    
    return portfolio, v2

def map_and_merge(dfs_list):
    """Takes a list of dataframes (1 or 3), normalizes their columns, and merges them securely"""
    unified_df = pd.DataFrame()
    
    col_mapping = {
        'buyer': 'buyer', 'company': 'buyer',
        'account_status': 'account_status', 'utilization_status': 'account_status',
        'am': 'am_names', 'am_name': 'am_names',
        'facility_size': 'facility_size',
        'outstanding_balance': 'outstanding_balance', 'ob': 'outstanding_balance',
        'last_disbursed_date': 'last_disbursed_date',
        'due_date_of_invoice': 'due_date_invoice',
        'settlement_date': 'settlement_date',
        'payment_total_usd': 'payment_total_usd',
        'am_email': 'am_view2'
    }
    
    mapped_dfs = []
    for i, raw_df in enumerate(dfs_list):
        # Normalize columns using fuzzy mapping
        df = raw_df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns=col_mapping)
        
        # Safely resolve duplicated column names by combining them (prioritizing non-nulls)
        if df.columns.duplicated().any():
            new_df = pd.DataFrame()
            for col in df.columns.unique():
                if isinstance(df[col], pd.DataFrame):
                    s = df[col].iloc[:, 0]
                    for j in range(1, df[col].shape[1]):
                        s = s.combine_first(df[col].iloc[:, j])
                    new_df[col] = s
                else:
                    new_df[col] = df[col]
            df = new_df
            
        mapped_dfs.append(df)
        
    if len(mapped_dfs) == 1:
        # Single sheet upload
        unified_df = mapped_dfs[0]
    elif len(mapped_dfs) >= 3:
        # 3 separate sheets or files
        m, v1, v2 = mapped_dfs[0], mapped_dfs[1], mapped_dfs[2]
        unified_df = pd.merge(m, v1, on='buyer', how='outer', suffixes=('', '_drop1'))
        
        # Combine overlapping columns safely
        if 'outstanding_balance_drop1' in unified_df.columns:
            unified_df['outstanding_balance'] = unified_df['outstanding_balance_drop1'].combine_first(unified_df.get('outstanding_balance'))
        if 'am_names_drop1' in unified_df.columns:
            unified_df['am_names'] = unified_df.get('am_names').combine_first(unified_df['am_names_drop1'])
            
        unified_df = pd.merge(unified_df, v2, on='buyer', how='outer', suffixes=('', '_drop2'))
        
        # Clean drops
        unified_df = unified_df.loc[:, ~unified_df.columns.str.endswith('_drop1')]
        unified_df = unified_df.loc[:, ~unified_df.columns.str.endswith('_drop2')]
    
    return apply_business_logic(unified_df)

@st.cache_data
def process_data_payload(payload_type, payloads):
    try:
        if payload_type == "single":
            df = pd.read_csv(payloads[0]) if payloads[0].name.lower().endswith('.csv') else pd.read_excel(payloads[0])
            return map_and_merge([df])
        elif payload_type == "combined_excel":
            xl = pd.ExcelFile(payloads[0])
            dfs = [xl.parse(s) for s in xl.sheet_names]
            return map_and_merge(dfs)
        elif payload_type == "three_files":
            dfs = []
            for f in payloads:
                dfs.append(pd.read_csv(f) if f.name.lower().endswith('.csv') else pd.read_excel(f))
            return map_and_merge(dfs)
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None, None

# --- 2. SIDEBAR UPLOAD & FILTERS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3003/3003299.png", width=60)
    st.title("Data Ingestion")
    
    upload_mode = st.radio("Upload Mode:", ["Single Sheet / Combined File", "Three Separate Files"])
    portfolio = v2 = None
    
    if upload_mode == "Single Sheet / Combined File":
        single = st.file_uploader("Upload Data (Excel or CSV)", type=["xlsx", "csv"])
        if st.button("Process Data", use_container_width=True) and single:
            with st.spinner("Processing Business Logic..."):
                if single.name.lower().endswith('.csv'):
                    portfolio, v2 = process_data_payload("single", [single])
                else:
                    xl = pd.ExcelFile(single)
                    if len(xl.sheet_names) == 1:
                        portfolio, v2 = process_data_payload("single", [single])
                    else:
                        portfolio, v2 = process_data_payload("combined_excel", [single])
                
                if portfolio is not None:
                    st.session_state['portfolio'] = portfolio
                    st.session_state['v2'] = v2
                
    else:
        f_m = st.file_uploader("Upload Masterdata", type=["xlsx", "csv"])
        f_v1 = st.file_uploader("Upload View 1", type=["xlsx", "csv"])
        f_v2 = st.file_uploader("Upload View 2", type=["xlsx", "csv"])
        
        if st.button("Process Data", use_container_width=True) and f_m and f_v1 and f_v2:
            with st.spinner("Processing Business Logic..."):
                portfolio, v2 = process_data_payload("three_files", [f_m, f_v1, f_v2])
                if portfolio is not None:
                    st.session_state['portfolio'] = portfolio
                    st.session_state['v2'] = v2

# --- 3. MAIN DASHBOARD ---
if 'portfolio' in st.session_state:
    df_port = st.session_state['portfolio']
    df_v2 = st.session_state['v2']
    
    st.title("AM Portfolio Command Center")
    
    # Global Filters applied to Portfolio Base
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        am_options = sorted([str(x) for x in df_port['am_names'].unique() if pd.notna(x)])
        am_filter = st.multiselect("Filter by AM Name:", options=am_options, default=[])
    with col_f2:
        status_options = sorted([str(x) for x in df_port['account_status'].unique() if pd.notna(x)])
        status_filter = st.multiselect("Filter by Account Status:", options=status_options, default=[])
        
    filtered_port = df_port.copy()
    if am_filter: filtered_port = filtered_port[filtered_port['am_names'].isin(am_filter)]
    if status_filter: filtered_port = filtered_port[filtered_port['account_status'].isin(status_filter)]
    
    # Cascade filters to V2
    filtered_v2 = df_v2[df_v2['buyer'].isin(filtered_port['buyer'])].copy()

    # --- Section A: Top Level KPIs ---
    st.markdown("### 📈 Portfolio Overview")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='margin:0; font-size:16px; color:#888;'>Total Facility Size</h3>
            <h2 style='margin:0; color:#00C4B4;'>${filtered_port['facility_size'].sum():,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='margin:0; font-size:16px; color:#888;'>Total Outstanding</h3>
            <h2 style='margin:0; color:#00C4B4;'>${filtered_port['outstanding_balance'].sum():,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        avg_util = filtered_port['utilisation_pct'].mean()
        if pd.isna(avg_util): avg_util = 0
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='margin:0; font-size:16px; color:#888;'>Avg Utilisation</h3>
            <h2 style='margin:0; color:#00C4B4;'>{avg_util:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        alerts = filtered_port['alert_180_days'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='margin:0; font-size:16px; color:#888;'>180-Day Alerts</h3>
            <h2 style='margin:0; color:#FF4B4B;'>{alerts} Accounts</h2>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    # --- Single Page Layout Sections via Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢 Company-Level View (View 1 & Master)", 
        "🧾 Invoices & Repayments (View 2)", 
        "⚠️ Workable-Inactive (AM)", 
        "🏆 Top Accounts"
    ])
    
    with tab1:
        st.markdown("#### Portfolio Base & Utilisation")
        st.dataframe(
            filtered_port[['buyer', 'account_status', 'am_names', 'facility_size', 'outstanding_balance', 'utilisation_pct', 'utilisation_category', 'last_disbursed_date', 'alert_180_days']],
            use_container_width=True, hide_index=True
        )
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Utilisation Category Distribution")
            if not filtered_port.empty:
                fig = px.pie(filtered_port, names='utilisation_category', hole=0.4, color_discrete_sequence=['#FF4B4B', '#FFA500', '#00C4B4'])
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Post-Repayment Low Utilisation (< 50%)")
            low_post = filtered_port[filtered_port['low_util_post_repay_flag']]
            st.dataframe(low_post[['buyer', 'facility_size', 'outstanding_balance', 'deduped_repayment', 'post_repayment_util']], use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("#### Present-Month Rule (Current Calendar Month)")
        v2_present = filtered_v2[filtered_v2['present_month_rule']]
        st.dataframe(v2_present[['buyer', 'due_date_invoice', 'settlement_date', 'payment_total_usd', 'am_view2']], use_container_width=True, hide_index=True)
        
        st.markdown("#### Collect Amount Flags (Missing Settlement Date)")
        collect = filtered_v2[filtered_v2['collect_amount_flag']]
        st.dataframe(collect[['buyer', 'due_date_invoice', 'payment_total_usd', 'am_view2']], use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Dedicated View: Workable-Inactive (AM)")
        inactive_am = filtered_port[filtered_port['account_status'] == 'Workable-Inactive (AM)']
        st.metric("Total Inactive AM Accounts", len(inactive_am))
        st.dataframe(inactive_am[['buyer', 'am_names', 'facility_size', 'last_disbursed_date', 'alert_180_days']], use_container_width=True, hide_index=True)

    with tab4:
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Top 15 Accounts (By Facility Size)")
            top_15 = df_port.sort_values('facility_size', ascending=False).head(15)
            if not top_15.empty and top_15['facility_size'].sum() > 0:
                fig_bar = px.bar(top_15, x='facility_size', y='buyer', orientation='h', color='facility_size', color_continuous_scale='teal')
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No facility size data available.")
            
        with c4:
            st.markdown("#### Top 50 Team Direct (By Facility Size)")
            team_direct = df_port[df_port['account_status'] == 'Team Direct'].sort_values('facility_size', ascending=False).head(50)
            if not team_direct.empty:
                st.dataframe(team_direct[['buyer', 'facility_size', 'outstanding_balance', 'utilisation_pct']], use_container_width=True, hide_index=True)
            else:
                st.info("No 'Team Direct' accounts found in the dataset.")

else:
    st.markdown("""
        <div style="text-align: center; padding-top: 100px;">
            <h1 style="color: #00C4B4;">Welcome to the AM Portfolio Dashboard</h1>
            <p style="color: #888; font-size: 18px;">Please upload your data via the sidebar to generate the intelligence reports.</p>
        </div>
    """, unsafe_allow_html=True)