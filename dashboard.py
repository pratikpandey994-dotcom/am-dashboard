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

# --- 1. DATA PROCESSING (PHASE 3 LOGIC) ---
@st.cache_data
def process_data(m, v1, v2):
    # Masterdata Mapping
    m = m[['Buyer', 'Account_Status', 'AM']].rename(columns={'Buyer': 'buyer', 'AM': 'am_names'})
    m['company'] = m['buyer']
    m['account_status'] = m['Account_Status'].astype(str).str.replace(r'\s*-\s*', '-', regex=True).str.strip().str.title()
    m['account_status'] = m['account_status'].str.replace('(Am)', '(AM)', regex=False)
    valid_statuses = ['Workable-Active', 'Workable-Inactive (AM)', 'Workable-Temporarily Suspended', 'Team Direct']
    m = m[m['account_status'].isin(valid_statuses)]
    m = m.drop(columns=['Account_Status'])

    # View 1 Mapping
    v1 = v1[['company', 'Outstanding_Balance', 'Facility_Size', 'Last_Disbursed_Date']].rename(columns={
        'company': 'buyer',
        'Outstanding_Balance': 'outstanding_balance',
        'Facility_Size': 'facility_size',
        'Last_Disbursed_Date': 'last_disbursed_date'
    })

    # View 2 Mapping
    v2 = v2[['Buyer', 'due_date_of_invoice', 'settlement_date', 'payment_total_usd', 'AM_Email']].rename(columns={
        'Buyer': 'buyer',
        'due_date_of_invoice': 'due_date_invoice',
        'AM_Email': 'am_view2'
    })

    # Merge Portfolio
    portfolio = pd.merge(m, v1, on='buyer', how='left')
    portfolio['am_names'] = portfolio['am_names'].fillna('Unassigned')
    portfolio.loc[portfolio['am_names'] == '', 'am_names'] = 'Unassigned'
    
    portfolio['facility_size'] = pd.to_numeric(portfolio['facility_size'], errors='coerce').fillna(0)
    portfolio['outstanding_balance'] = pd.to_numeric(portfolio['outstanding_balance'], errors='coerce').fillna(0)
    
    portfolio['utilisation_pct'] = np.where(portfolio['facility_size'] > 0, (portfolio['outstanding_balance'] / portfolio['facility_size']) * 100, 0)
    portfolio['utilisation_category'] = pd.cut(portfolio['utilisation_pct'], bins=[-np.inf, 50, 70, np.inf], labels=['Low', 'Medium', 'High'], right=False)
    
    now = pd.Timestamp.now()
    portfolio['last_disbursed_date'] = pd.to_datetime(portfolio['last_disbursed_date'], errors='coerce')
    portfolio['alert_180_days'] = (now - portfolio['last_disbursed_date']).dt.days > 180

    # View 2 Logic
    v2['due_date_invoice'] = pd.to_datetime(v2['due_date_invoice'], errors='coerce')
    v2['settlement_date'] = pd.to_datetime(v2['settlement_date'], errors='coerce')
    v2['payment_total_usd'] = pd.to_numeric(v2['payment_total_usd'], errors='coerce').fillna(0)
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

def load_file(file):
    return pd.read_csv(file) if file.name.lower().endswith('.csv') else pd.read_excel(file)


# --- 2. SIDEBAR UPLOAD & FILTERS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3003/3003299.png", width=60)
    st.title("Data Ingestion")
    
    upload_mode = st.radio("Upload Mode:", ["Three Separate Files", "Single File (3 Sheets)"])
    portfolio = v2 = None
    
    if upload_mode == "Three Separate Files":
        f_m = st.file_uploader("Upload Masterdata", type=["xlsx", "csv"])
        f_v1 = st.file_uploader("Upload View 1", type=["xlsx", "csv"])
        f_v2 = st.file_uploader("Upload View 2", type=["xlsx", "csv"])
        
        if st.button("Process Data", use_container_width=True) and f_m and f_v1 and f_v2:
            with st.spinner("Processing Business Logic..."):
                portfolio, v2 = process_data(load_file(f_m), load_file(f_v1), load_file(f_v2))
                st.session_state['portfolio'] = portfolio
                st.session_state['v2'] = v2
                
    else:
        single = st.file_uploader("Upload Combined Excel", type=["xlsx"])
        if st.button("Process Data", use_container_width=True) and single:
            with st.spinner("Processing Business Logic..."):
                xl = pd.ExcelFile(single)
                if len(xl.sheet_names) >= 3:
                    portfolio, v2 = process_data(xl.parse(0), xl.parse(1), xl.parse(2))
                    st.session_state['portfolio'] = portfolio
                    st.session_state['v2'] = v2
                else:
                    st.error("Excel file must contain at least 3 sheets.")

# --- 3. MAIN DASHBOARD ---
if 'portfolio' in st.session_state:
    df_port = st.session_state['portfolio']
    df_v2 = st.session_state['v2']
    
    st.title("AM Portfolio Command Center")
    
    # Global Filters applied to Portfolio Base
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        am_filter = st.multiselect("Filter by AM Name:", options=sorted(df_port['am_names'].unique()), default=[])
    with col_f2:
        status_filter = st.multiselect("Filter by Account Status:", options=sorted(df_port['account_status'].unique()), default=[])
        
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
            fig_bar = px.bar(top_15, x='facility_size', y='buyer', orientation='h', color='facility_size', color_continuous_scale='teal')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
            
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
            <p style="color: #888; font-size: 18px;">Please upload your Masterdata, View 1, and View 2 files via the sidebar to generate the intelligence reports.</p>
        </div>
    """, unsafe_allow_html=True)