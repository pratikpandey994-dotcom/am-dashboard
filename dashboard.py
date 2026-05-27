import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import os
import glob

# Set page config
st.set_page_config(page_title="Account Manager Performance Dashboard", layout="wide")

# --- UTILITY FUNCTIONS ---

def find_most_recent_excel(directory):
    """Finds the latest Excel file in the specified directory."""
    list_of_files = glob.glob(os.path.join(directory, "*.xlsx"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def dynamic_map_columns(df):
    """Fuzzy match columns to standard business terms."""
    cols = df.columns.tolist()
    mapping = {}
    
    # Define candidates for common entities
    candidates = {
        "AM": ["am_name", "owner", "account_manager", "rm_name", "relationship_manager", "assigned_to", "am_email"],
        "Account": ["company", "buyer", "seller", "account_name", "client", "customer"],
        "Revenue": ["realised revenue", "booked_revenue", "revenue", "mrr", "arr", "total_fees"],
        "Pipeline": ["outstanding_balance", "outstanding", "facility_size", "total advanced", "origination"],
        "Status": ["utilization_status", "status", "stage", "state"],
        "Date": ["date", "created", "closed", "updated", "disbursed", "settlement", "invoice"]
    }
    
    for key, choices in candidates.items():
        # Try to find the best match in the dataframe columns
        matches = []
        for choice in choices:
            # For AM, we want to be more specific to prefer Name over Email if both exist
            match, score = process.extractOne(choice, cols)
            if score > 85:
                # If we are looking for AM and we found a 'name' column, prioritize it
                if key == "AM" and "name" in match.lower():
                    matches.insert(0, match) # Put name at the front
                else:
                    matches.append(match)
        
        # Take the best unique match
        if matches:
            # For AM, specifically avoid picking the email if a name is available
            if key == "AM":
                names = [m for m in matches if "name" in m.lower() or "manager" in m.lower() or "owner" in m.lower()]
                if names:
                    mapping[key] = [names[0]]
                else:
                    mapping[key] = [matches[0]]
            else:
                mapping[key] = [matches[0]]
            
    return mapping

# --- UI LAYOUT ---

st.title("📊 Account Manager Dashboard")
st.markdown("Upload your daily Excel export to refresh the views.")

# 1. File Upload / Auto-detect
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Daily Export", type=["xlsx"])
    
    # Optional: Suggest latest from Downloads folder
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    suggested_file = find_most_recent_excel(downloads_path)
    
    if suggested_file and not uploaded_file:
        if st.button(f"Load Latest: {os.path.basename(suggested_file)}"):
            uploaded_file = suggested_file

# 2. Processing
if uploaded_file:
    try:
        # Load data
        df = pd.read_excel(uploaded_file)
        mapping = dynamic_map_columns(df)
        
        # --- Sidebar Filters ---
        st.sidebar.divider()
        st.sidebar.header("Filters")
        
        selected_am = "All"
        if "AM" in mapping:
            am_col = mapping["AM"][0]
            am_list = ["All"] + sorted(df[am_col].dropna().unique().tolist())
            selected_am = st.sidebar.selectbox("Filter by Account Manager", am_list)
        
        # Apply filtering
        plot_df = df.copy()
        if selected_am != "All":
            plot_df = plot_df[plot_df[am_col] == selected_am]

        # --- Dashboard Metrics (KPIs) ---
        col1, col2, col3, col4 = st.columns(4)
        
        # Total Accounts
        if "Account" in mapping:
            acc_col = mapping["Account"][0]
            col1.metric("Total Accounts", plot_df[acc_col].nunique())
            
        # Total Revenue
        if "Revenue" in mapping:
            rev_col = mapping["Revenue"][0]
            col2.metric("Total Revenue", f"${plot_df[rev_col].sum():,.0f}")
            
        # Total Pipeline
        if "Pipeline" in mapping:
            pipe_col = mapping["Pipeline"][0]
            col3.metric("Total Exposure", f"${plot_df[pipe_col].sum():,.0f}")
            
        # At Risk
        if "Status" in mapping:
            stat_col = mapping["Status"][0]
            risk_count = len(plot_df[plot_df[stat_col].str.contains("suspended|risk|overdue", case=False, na=False)])
            col4.metric("Accounts at Risk", risk_count, delta_color="inverse")

        # --- Visualizations ---
        st.divider()
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.subheader("Portfolio Distribution by Status")
            if "Status" in mapping:
                fig = px.pie(plot_df, names=mapping["Status"][0], hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No 'Status' column detected.")

        with row1_col2:
            st.subheader("Top Accounts by Exposure")
            if "Account" in mapping and "Pipeline" in mapping:
                top_n = plot_df.groupby(mapping["Account"][0])[mapping["Pipeline"][0]].sum().nlargest(10).reset_index()
                fig = px.bar(top_n, x=mapping["Pipeline"][0], y=mapping["Account"][0], orientation='h',
                             labels={mapping["Pipeline"][0]: "Exposure ($)"}, color=mapping["Pipeline"][0])
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        
        # 3. Time Series / Growth
        st.subheader("Revenue Trend over Time")
        date_cols = [c for c in df.columns if any(x in c.lower() for x in ["date", "time", "created"])]
        if date_cols and "Revenue" in mapping:
            d_col = date_cols[0]
            plot_df[d_col] = pd.to_datetime(plot_df[d_col], errors='coerce')
            trend_df = plot_df.resample('ME', on=d_col)[mapping["Revenue"][0]].sum().reset_index() # Updated 'M' to 'ME' for pandas 3.0
            fig = px.area(trend_df, x=d_col, y=mapping["Revenue"][0], markers=True,
                          title=f"Monthly Revenue Growth ({d_col})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload a file with date columns to see revenue trends.")

        # 4. Raw Data Preview
        with st.expander("View Raw Data"):
            st.dataframe(plot_df)

    except Exception as e:
        st.error(f"Critical Error: Could not parse the Excel file. Details: {e}")
        st.info("Ensure the Excel file is not password protected and has headers in the first row.")
else:
    st.info("Waiting for file upload. Use the sidebar to upload your daily export.")
