import streamlit as st
import pandas as pd
import plotly.express as px
from fuzzywuzzy import process

# Set page config
st.set_page_config(page_title="Smart Analytics Dashboard", layout="wide")

# --- UTILITY FUNCTIONS ---

def dynamic_map_columns(df):
    """Fuzzy match columns to standard business terms."""
    cols = df.columns.tolist()
    mapping = {}
    
    # Expanded candidates for more robustness
    candidates = {
        "AM": ["am_name", "owner", "account_manager", "rm_name", "relationship_manager", "assigned_to", "am_email"],
        "Account": ["company", "buyer", "seller", "account_name", "client", "customer"],
        "Revenue": ["realised revenue", "booked_revenue", "revenue", "mrr", "arr", "total_fees", "amount", "sales"],
        "Pipeline": ["outstanding_balance", "outstanding", "facility_size", "total advanced", "origination", "pipeline", "forecast"],
        "Status": ["utilization_status", "status", "stage", "state"],
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
st.markdown("Upload **any** Excel or CSV file. The system will auto-detect the schema, fix errors, and build visualisations.")

with st.sidebar:
    st.header("Data Source")
    # Added CSV support
    uploaded_file = st.file_uploader("Upload Data File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("Could not read the file. Please ensure it's a valid CSV or Excel document with data.")
    else:
        mapping = dynamic_map_columns(df)
        
        # --- TABBED INTERFACE ---
        tab1, tab2, tab3 = st.tabs(["🎯 Business View (AM)", "🔍 Smart Auto-Discovery", "📋 Raw Data"])
        
        # TAB 1: Fixed Business Logic
        with tab1:
            st.markdown("### Account Manager Portfolio")
            if not mapping:
                st.info("Could not auto-detect standard business columns (AM, Account, Revenue). Try the 'Smart Auto-Discovery' tab!")
            else:
                # Setup Filter
                plot_df = df.copy()
                selected_am = "All"
                if "AM" in mapping:
                    am_col = mapping["AM"][0]
                    # Drop NA and convert to string for safe filtering
                    am_list = ["All"] + sorted([str(x) for x in df[am_col].dropna().unique()])
                    selected_am = st.selectbox("Filter by Account Manager", am_list)
                    if selected_am != "All":
                        plot_df = plot_df[plot_df[am_col].astype(str) == selected_am]

                # KPIs
                m1, m2, m3, m4 = st.columns(4)
                if "Account" in mapping:
                    acc_col = mapping["Account"][0]
                    m1.metric("Total Unique Accounts", plot_df[acc_col].nunique())
                    
                if "Revenue" in mapping:
                    rev_col = mapping["Revenue"][0]
                    # Force numeric to prevent text crashing the sum
                    revenue_sum = pd.to_numeric(plot_df[rev_col], errors='coerce').sum()
                    m2.metric("Total Revenue", f"${revenue_sum:,.0f}")
                    
                if "Pipeline" in mapping:
                    pipe_col = mapping["Pipeline"][0]
                    pipeline_sum = pd.to_numeric(plot_df[pipe_col], errors='coerce').sum()
                    m3.metric("Total Pipeline/Exposure", f"${pipeline_sum:,.0f}")
                
                # BUG FIX: Accurate Risk Calculation
                if "Status" in mapping:
                    stat_col = mapping["Status"][0]
                    # Create a boolean mask for rows at risk
                    risk_mask = plot_df[stat_col].astype(str).str.contains("suspended|risk|overdue", case=False, na=False)
                    
                    if "Account" in mapping:
                        # Count UNIQUE accounts that have a risk flag, not total transactions
                        risk_count = plot_df[risk_mask][acc_col].nunique()
                    else:
                        risk_count = risk_mask.sum()
                        
                    m4.metric("Accounts at Risk", risk_count, delta_color="inverse")

                st.divider()
                
                # Visualizations
                c1, c2 = st.columns(2)
                with c1:
                    if "Status" in mapping:
                        st.subheader("Distribution by Status")
                        if "Account" in mapping:
                            # Group by unique accounts to prevent transaction volume from skewing the donut
                            stat_df = plot_df.groupby(stat_col)[acc_col].nunique().reset_index()
                            fig1 = px.pie(stat_df, names=stat_col, values=acc_col, hole=0.4)
                        else:
                            fig1 = px.pie(plot_df, names=stat_col, hole=0.4)
                        st.plotly_chart(fig1, use_container_width=True)
                
                with c2:
                    if "Account" in mapping and "Pipeline" in mapping:
                        st.subheader("Top Accounts")
                        plot_df[pipe_col] = pd.to_numeric(plot_df[pipe_col], errors='coerce')
                        top_n = plot_df.groupby(acc_col)[pipe_col].sum().nlargest(10).reset_index()
                        fig2 = px.bar(top_n, x=pipe_col, y=acc_col, orientation='h')
                        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig2, use_container_width=True)

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
