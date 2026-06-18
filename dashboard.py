from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


KEEP_STATUSES = {
    "Non Workable",
    "Workable - Active",
    "Workable - InActive BD",
    "Workable - InActive AM",
    "Workable - Temp Suspended",
}

STATE_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

THEME_PALETTE = [
    "#157f3b", "#3b82f6", "#0d9488", "#c47a00", "#b42318", # Bases (Green, Blue, Teal, Orange, Red)
    "#34d399", "#60a5fa", "#2dd4bf", "#fbbf24", "#f87171", # Lights
    "#064e3b", "#1e3a8a", "#115e59", "#78350f", "#7f1d1d", # Darks
    "#a7f3d0", "#dbeafe", "#ccfbf1", "#fef3c7", "#fecaca"  # Lighters
]
TEAL_SCALE = ["#ccfbf1", "#2dd4bf", "#0d9488", "#115e59"]
BLUE_SCALE = ["#dbeafe", "#60a5fa", "#3b82f6", "#1e3a8a"]
GREEN_SCALE = ["#a7f3d0", "#34d399", "#157f3b", "#064e3b"]


@dataclass(frozen=True)
class LoadedData:
    master: Optional[pd.DataFrame]
    view1: Optional[pd.DataFrame]
    view2: Optional[pd.DataFrame]
    flexible: Optional[pd.DataFrame]
    source_mode: str
    mode: str


st.set_page_config(page_title="AM Portfolio Dashboard", layout="wide", initial_sidebar_state="collapsed")


def inject_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1f2937;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1580px;
        }
        
        h1 {
            font-weight: 700;
            color: #111827;
            letter-spacing: -0.025em;
            margin-bottom: 0.2rem;
        }
        
        h2, h3 {
            font-weight: 600;
            color: #374151;
            letter-spacing: -0.01em;
        }
        
        /* Metric Box Styling */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #f3f4f6;
            border-radius: 12px;
            padding: 20px 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease-in-out;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        }
        
        div[data-testid="stMetricLabel"] {
            color: #6b7280;
            font-size: 0.875rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }
        
        div[data-testid="stMetricValue"] {
            color: #111827;
            font-size: 1.875rem;
            font-weight: 700;
        }
        
        .section-note {
            color: #9ca3af;
            font-size: 0.875rem;
            margin-top: -0.5rem;
            margin-bottom: 1.5rem;
            font-style: italic;
        }
        
        /* Expander & Container Styling */
        div[data-testid="stExpander"] {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            background: #ffffff;
            overflow: hidden;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #6b7280;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            color: #0d9488 !important;
            border-bottom-color: #0d9488 !important;
            font-weight: 600 !important;
        }
        
        /* Divider Styling */
        hr {
            margin: 2rem 0;
            border-color: #f3f4f6;
        }

        /* Dataframe Styling */
        .stDataFrame {
            border: 1px solid #f3f4f6;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_name(value: object) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def normalize_status(series: pd.Series) -> pd.Series:
    status_map = {
        "non workable": "Non Workable",
        "non-workable": "Non Workable",
        "nonworkable": "Non Workable",
        "workable - active": "Workable - Active",
        "workable active": "Workable - Active",
        "workable-active": "Workable - Active",
        "workable - inactive bd": "Workable - InActive BD",
        "workable inactive bd": "Workable - InActive BD",
        "workable-inactive bd": "Workable - InActive BD",
        "workable - inactive am": "Workable - InActive AM",
        "workable inactive am": "Workable - InActive AM",
        "workable-inactive am": "Workable - InActive AM",
        "workable - inactive": "Workable - InActive AM",
        "workable inactive": "Workable - InActive AM",
        "workable- inactive": "Workable - InActive AM",
        "workable - temp suspended": "Workable - Temp Suspended",
        "workable temp suspended": "Workable - Temp Suspended",
        "workable-temp suspended": "Workable - Temp Suspended",
        "workable-temporarily suspended": "Workable - Temp Suspended",
        "temp suspended": "Workable - Temp Suspended",
        "suspended": "Workable - Temp Suspended"
    }
    return series.astype("string").str.strip().apply(
        lambda x: status_map.get(str(x).lower(), "Unknown" if str(x).lower() == "nan" else x)
    )


def clean_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    exact = {str(col): col for col in df.columns}
    normalized = {normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def require_columns(df: pd.DataFrame, columns: Dict[str, Optional[str]], sheet_name: str) -> None:
    missing = [logical for logical, raw in columns.items() if raw is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{sheet_name} is missing required column(s): {joined}")


def workbook_sheets(file) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(file)
    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}


def classify_sheet(name: str, df: pd.DataFrame) -> Optional[str]:
    normalized_cols = {normalize_name(col) for col in df.columns}
    normalized_name = normalize_name(name)

    if {"account status", "buyer", "ob"}.issubset(normalized_cols):
        return "master"
    if {"company", "outstanding balance", "am name"}.issubset(normalized_cols):
        return "view1"
    if {"invoice id", "settlement date", "payment total usd"}.issubset(normalized_cols):
        return "view2"

    if "master" in normalized_name:
        return "master"
    if "view 1" in normalized_name or "company" in normalized_name:
        return "view1"
    if "view 2" in normalized_name or "invoice" in normalized_name or "repayment" in normalized_name:
        return "view2"
    return None


@st.cache_data(show_spinner=False, persist="disk")
def get_excel_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))

def load_uploaded_data(master_file, invoice_file) -> Optional[LoadedData]:
    if master_file and invoice_file:
        master_df = get_excel_data(master_file.getvalue())
        invoice_df = get_excel_data(invoice_file.getvalue())
        return LoadedData(
            master=master_df,
            view1=master_df,
            view2=invoice_df,
            flexible=None,
            source_mode="Two uploaded files",
            mode="full",
        )
    return None


@st.cache_data(show_spinner=False, persist="disk")
def build_logic(master: pd.DataFrame, view1: pd.DataFrame, view2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master_cols = {
        "Buyer": first_existing(master, ["Buyer", "IMPORTER_NAME", "IMPORTER_COMPANY", "CP_COMPANY"]),
        "Account_Status": first_existing(master, ["Account_Status", "Account Status", "ACCOUNT_STATUS"]),
        "User_State": first_existing(master, ["User_State", "User State", "USER_STATE"]),
        "AM": first_existing(master, ["AM"]),
        "Team": first_existing(master, ["Team", "POD_MANAGER"]),
        "Facility_Size": first_existing(master, ["Facility_Size", "Facility Size", "FACILITY_SIZE", "TOTAL_LIMIT"]),
        "OB": first_existing(master, ["OB", "Outstanding Balance", "OUTSTANDING_ADVANCE_BALANCE_USD"]),
        "Last_Disbursed_Date": first_existing(master, ["Last_Disbursed_Date", "Last Disbursed Date", "LAST_DISBURSED_DATE"]),
    }
    master_opt_cols = {
        "Industry": first_existing(master, ["Industry", "AIRTABLE_INDUSTRY", "INDUSTRY"]),
    }
    view1_cols = {
        "company": first_existing(view1, ["company", "Buyer", "Account", "IMPORTER_NAME", "IMPORTER_COMPANY", "CP_COMPANY"]),
        "AM_Name": first_existing(view1, ["AM_Name", "AM Name", "AM"]),
        "Facility_Size": first_existing(view1, ["Facility_Size", "Facility Size", "Limit", "FACILITY_SIZE", "TOTAL_LIMIT"]),
        "Outstanding_Balance": first_existing(view1, ["Outstanding_Balance", "Outstanding Balance", "OB", "Outstanding", "Balance", "OUTSTANDING_ADVANCE_BALANCE_USD"]),
        "Last_Disbursed_Date": first_existing(view1, ["Last_Disbursed_Date", "Last Disbursed Date", "Disbursed Date", "LAST_DISBURSED_DATE"]),
    }
    view2_cols = {
        "Buyer": first_existing(view2, ["Buyer", "Account", "company", "IMPORTER_COMPANY", "CP_COMPANY"]),
        "AM_Email": first_existing(view2, ["AM_Email", "AM Email", "AM"]),
        "due_date_of_invoice": first_existing(view2, ["due_date_of_invoice", "due date of invoice", "Due Date", "DUE_DATE"]),
        "settlement_date": first_existing(view2, ["settlement_date", "settlement date", "Settlement Date", "SETTLEMENT_DATE"]),
        "payment_total_usd": first_existing(view2, ["payment_total_usd", "payment total usd", "payment", "Amount", "MARGIN_RECEIVED_USD", "INVOICE_VALUE_USD"]),
        "disbursed_date": first_existing(view2, ["disbursed_date", "disbursed date", "disbursement date", "Disbursed Date", "FIRST_ADVANCE_DATE", "INVOICE_DATE"]),
        "total_advanced": first_existing(view2, ["total_advanced", "total advanced", "Total Advanced", "Origination", "TOTAL_ADVANCED"]),
    }
    view2_opt_cols = {
        "invoice_id": first_existing(view2, ["INVOICE_ID", "Invoice ID", "invoice id"]),
        "exporter_company": first_existing(view2, ["EXPORTER_COMPANY", "Exporter Company", "exporter", "Seller", "EXPORTER_USER_ID"]),
        "exporter_country": first_existing(view2, ["EXPORTER_COUNTRY", "Exporter Country", "BUYER_COUNTRY"]),
        "margin_received_date": first_existing(view2, ["MARGIN_RECEIVED_DATE", "Margin Received Date", "margin_received_date"]),
    }
    master_req = {k: v for k, v in master_cols.items() if k != "User_State"}
    require_columns(master, master_req, "Masterdata")
    require_columns(view1, view1_cols, "View 1")
    require_columns(view2, view2_cols, "View 2")

    if master_cols.get("Account_Status") and master_cols["Account_Status"] in master.columns:
        master[master_cols["Account_Status"]] = normalize_status(master[master_cols["Account_Status"]])

    master_l = master.assign(
        buyer=master[master_cols["Buyer"]].astype("string").str.strip(),
        buyer_key=clean_key(master[master_cols["Buyer"]]),
        account_status=normalize_status(master[master_cols["Account_Status"]]),
        user_state=master[master_cols["User_State"]].astype("string").str.strip() if master_cols.get("User_State") and master_cols["User_State"] in master.columns else "Unknown",
        am_master=master[master_cols["AM"]].astype("string").str.strip(),
        team=master[master_cols["Team"]].astype("string").str.strip(),
        facility_size_master=to_number(master[master_cols["Facility_Size"]]),
        outstanding_balance_master=to_number(master[master_cols["OB"]]),
        last_disbursed_date_master=to_date(master[master_cols["Last_Disbursed_Date"]]),
        industry=master[master_opt_cols["Industry"]].astype("string").str.strip() if master_opt_cols.get("Industry") else "Unknown",
    )
    # Dedup Master by buyer_key and keep original columns
    agg_dict = {col: "first" for col in master.columns}
    agg_dict.update({
        "buyer": "first",
        "account_status": "first",
        "user_state": "first",
        "am_master": "first",
        "team": "first",
        "industry": "first",
        "facility_size_master": "max",
        "outstanding_balance_master": "max",
        "last_disbursed_date_master": "max",
    })
    master_l = master_l.groupby("buyer_key", as_index=False).agg(agg_dict)
    master_l = master_l[master_l["account_status"].isin(KEEP_STATUSES)].copy()

    view1_l = view1.assign(
        buyer_key=clean_key(view1[view1_cols["company"]]),
        company=view1[view1_cols["company"]].astype("string").str.strip(),
        am_view1=view1[view1_cols["AM_Name"]].astype("string").str.strip(),
        facility_size_view1=to_number(view1[view1_cols["Facility_Size"]]),
        outstanding_balance_view1=to_number(view1[view1_cols["Outstanding_Balance"]]),
        last_disbursed_date_view1=to_date(view1[view1_cols["Last_Disbursed_Date"]]),
    )
    # Dedup View 1 by buyer_key
    view1_l = view1_l.groupby("buyer_key", as_index=False).agg({
        "company": "first",
        "am_view1": "first",
        "facility_size_view1": "max",
        "outstanding_balance_view1": "max",
        "last_disbursed_date_view1": "max",
    })

    accounts = master_l.merge(view1_l, on="buyer_key", how="left")
    accounts["company"] = accounts["company"].fillna(accounts["buyer"])
    accounts["am_names"] = (
        accounts["am_master"]
        .replace("", pd.NA)
        .fillna(accounts["am_view1"])
        .replace("", pd.NA)
        .fillna("Unassigned")
    )
    accounts["facility_size"] = (
        accounts["facility_size_view1"]
        .where(accounts["facility_size_view1"].notna(), accounts["facility_size_master"])
        .fillna(0)
    )
    accounts["outstanding_balance"] = (
        accounts["outstanding_balance_view1"]
        .where(accounts["outstanding_balance_view1"].notna(), accounts["outstanding_balance_master"])
        .fillna(0)
    )
    accounts["last_disbursed_date"] = accounts["last_disbursed_date_view1"].where(
        accounts["last_disbursed_date_view1"].notna(),
        accounts["last_disbursed_date_master"],
    )

    facility_denominator = accounts["facility_size"].where(accounts["facility_size"] > 0)
    accounts["utilisation_pct"] = (accounts["outstanding_balance"].div(facility_denominator) * 100).fillna(0)
    accounts["utilisation_pct"] = accounts["utilisation_pct"].replace([np.inf, -np.inf], 0).fillna(0)
    accounts["utilisation_category"] = np.select(
        [accounts["utilisation_pct"] >= 70, accounts["utilisation_pct"] >= 50],
        ["High", "Medium"],
        default="Low",
    )

    today = pd.Timestamp.today().normalize()
    # Handle NaT by setting to a very large number for comparison
    accounts["days_since_last_disbursed"] = (today - accounts["last_disbursed_date"]).dt.days.fillna(9999).astype(int)
    
    # Exclusive aging buckets
    accounts["alert_120_days"] = (accounts["days_since_last_disbursed"] > 120) & (accounts["days_since_last_disbursed"] <= 150)
    accounts["alert_150_days"] = (accounts["days_since_last_disbursed"] > 150) & (accounts["days_since_last_disbursed"] <= 180)
    accounts["alert_180_days"] = (accounts["days_since_last_disbursed"] > 180) & (accounts["days_since_last_disbursed"] <= 356)
    accounts["alert_356_days"] = (accounts["days_since_last_disbursed"] > 356) & (accounts["days_since_last_disbursed"] < 9999)

    view2_l = view2.assign(
        buyer=view2[view2_cols["Buyer"]].astype("string").str.strip(),
        buyer_key=clean_key(view2[view2_cols["Buyer"]]),
        am_view2=view2[view2_cols["AM_Email"]].astype("string").str.strip(),
        due_date_invoice=to_date(view2[view2_cols["due_date_of_invoice"]]),
        settlement_date=to_date(view2[view2_cols["settlement_date"]]),
        disbursed_date=to_date(view2[view2_cols["disbursed_date"]]),
        payment_total_usd=to_number(view2[view2_cols["payment_total_usd"]]),
        total_advanced=to_number(view2[view2_cols["total_advanced"]]),
        invoice_id=view2[view2_opt_cols["invoice_id"]].astype("string").str.strip() if view2_opt_cols.get("invoice_id") else pd.NA,
        exporter_company=view2[view2_opt_cols["exporter_company"]].astype("string").str.strip() if view2_opt_cols.get("exporter_company") else pd.NA,
        exporter_country=view2[view2_opt_cols["exporter_country"]].astype("string").str.strip() if view2_opt_cols.get("exporter_country") else pd.NA,
        margin_received_date=to_date(view2[view2_opt_cols["margin_received_date"]]) if view2_opt_cols.get("margin_received_date") else pd.NaT,
    )

    # Invoices & Repayments: cover current and previous month for better visibility
    lookback_date = (today - pd.DateOffset(months=1)).replace(day=1)
    present_month_mask = (
        (view2_l["due_date_invoice"] >= lookback_date)
        | (view2_l["settlement_date"] >= lookback_date)
        | (view2_l["disbursed_date"] >= lookback_date)
    )
    view2_present = view2_l[present_month_mask].copy()
    view2_present["collect_amount"] = view2_present["settlement_date"].isna()

    repayments_dedup = (
        view2_present.dropna(subset=["settlement_date"])
        .groupby(["buyer_key", "buyer", "settlement_date"], as_index=False)["payment_total_usd"]
        .sum()
        .rename(columns={"payment_total_usd": "deduped_repayment"})
    )

    repayment_by_buyer = repayments_dedup.groupby("buyer_key", as_index=False)["deduped_repayment"].sum()
    accounts = accounts.merge(repayment_by_buyer, on="buyer_key", how="left")
    accounts["deduped_repayment"] = accounts["deduped_repayment"].fillna(0)
    accounts["adjusted_outstanding"] = accounts["outstanding_balance"] - accounts["deduped_repayment"]
    facility_denominator = accounts["facility_size"].where(accounts["facility_size"] > 0)
    accounts["post_repayment_util"] = (accounts["adjusted_outstanding"].div(facility_denominator) * 100).fillna(0)
    accounts["post_repayment_util"] = accounts["post_repayment_util"].replace([np.inf, -np.inf], 0).fillna(0)
    accounts["low_utilisation_after_repayment"] = accounts["post_repayment_util"] < 50

    # OB Trend: Past 6 months
    trend_data = []
    for i in range(6, -1, -1):
        d = (today - pd.DateOffset(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
        # Outstanding at end of d = Sum(total_advanced where disbursed <= d) - Sum(payment where settlement <= d)
        advances = view2_l[view2_l["disbursed_date"] <= d]["total_advanced"].sum()
        settlements = view2_l[view2_l["settlement_date"] <= d]["payment_total_usd"].sum()
        trend_data.append({"Month": d.strftime("%b %Y"), "Outstanding Balance": max(0, advances - settlements)})
    ob_trend = pd.DataFrame(trend_data)

    keep_account_cols = [
        "buyer",
        "buyer_key",
        "company",
        "account_status",
        "user_state",
        "team",
        "industry",
        "am_names",
        "outstanding_balance",
        "facility_size",
        "last_disbursed_date",
        "days_since_last_disbursed",
        "alert_120_days",
        "alert_150_days",
        "alert_180_days",
        "alert_356_days",
        "utilisation_pct",
        "utilisation_category",
        "deduped_repayment",
        "adjusted_outstanding",
        "post_repayment_util",
        "low_utilisation_after_repayment",
    ]
    # Append any remaining unmapped columns from the raw master sheet
    extra_cols = [c for c in master.columns if c not in keep_account_cols and c not in ["buyer", "buyer_key", "company", "account_status", "team", "am_names"]]
    final_cols = keep_account_cols + extra_cols
    return accounts[final_cols], view2_present, repayments_dedup, ob_trend, view2_l


def choose_default(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    found = first_existing(df, candidates)
    return str(found) if found is not None else "(None)"


def column_selector(label: str, df: pd.DataFrame, candidates: Iterable[str], required: bool = False) -> Optional[str]:
    options = ["(None)"] + [str(col) for col in df.columns]
    default = choose_default(df, candidates)
    index = options.index(default) if default in options else 0
    help_text = "Required for account-level calculations." if required else None
    selected = st.selectbox(label, options, index=index, help=help_text)
    return None if selected == "(None)" else selected


def render_flexible_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    st.caption("Map the uploaded sheet columns into the dashboard fields.")
    row1 = st.columns(4)
    with row1[0]:
        buyer = column_selector("Buyer / Account", df, ["Buyer", "company", "Account", "Company", "IMPORTER_NAME", "IMPORTER_COMPANY", "CP_COMPANY"], required=True)
    with row1[1]:
        account_status = column_selector("Account Status", df, ["Account_Status", "Account Status", "Status", "Stage", "ACCOUNT_STATUS", "USER_UTILIZATION_STATUS"])
    with row1[2]:
        am_names = column_selector("AM Name", df, ["AM", "AM_Name", "AM Name", "Account Manager", "Owner"])
    with row1[3]:
        team = column_selector("Team", df, ["Team", "POD_MANAGER"])

    row2 = st.columns(4)
    with row2[0]:
        company = column_selector("Company", df, ["company", "Company", "Buyer", "IMPORTER_NAME", "IMPORTER_COMPANY", "CP_COMPANY"])
    with row2[1]:
        outstanding_balance = column_selector(
            "Outstanding Balance",
            df,
            ["Outstanding_Balance", "Outstanding Balance", "OB", "Outstanding", "OUTSTANDING_ADVANCE_BALANCE_USD"],
        )
    with row2[2]:
        facility_size = column_selector("Facility Size", df, ["Facility_Size", "Facility Size", "Limit", "FACILITY_SIZE", "TOTAL_LIMIT"])
    with row2[3]:
        last_disbursed_date = column_selector(
            "Last Disbursed Date",
            df,
            ["Last_Disbursed_Date", "Last Disbursed Date", "disbursed_date", "Disbursed Date", "LAST_DISBURSED_DATE"],
        )

    row3 = st.columns(4)
    with row3[0]:
        due_date_invoice = column_selector(
            "Due Date of Invoice",
            df,
            ["due_date_of_invoice", "Due Date of Invoice", "due date", "Due Date", "DUE_DATE"],
        )
    with row3[1]:
        settlement_date = column_selector("Settlement Date", df, ["settlement_date", "Settlement Date", "SETTLEMENT_DATE"])
    with row3[2]:
        payment_total_usd = column_selector(
            "Payment Total USD",
            df,
            ["payment_total_usd", "Payment Total USD", "payment", "Amount", "MARGIN_RECEIVED_USD", "INVOICE_VALUE_USD"],
        )
    with row3[3]:
        am_view2 = column_selector("View 2 AM / Email", df, ["AM_Email", "AM Email", "AM"])
    
    row4 = st.columns(4)
    with row4[0]:
        disbursed_date = column_selector("Disbursed Date", df, ["disbursed_date", "Disbursed Date", "FIRST_ADVANCE_DATE", "INVOICE_DATE"])
    with row4[1]:
        total_advanced = column_selector("Total Advanced", df, ["total_advanced", "Total Advanced", "TOTAL_ADVANCED"])

    return {
        "buyer": buyer,
        "account_status": account_status,
        "user_state": first_existing(df, ["User_State", "User State", "USER_STATE"]),
        "am_names": am_names,
        "team": team,
        "company": company,
        "outstanding_balance": outstanding_balance,
        "facility_size": facility_size,
        "last_disbursed_date": last_disbursed_date,
        "due_date_invoice": due_date_invoice,
        "settlement_date": settlement_date,
        "payment_total_usd": payment_total_usd,
        "am_view2": am_view2,
        "disbursed_date": disbursed_date,
        "total_advanced": total_advanced,
    }


@st.cache_data(show_spinner=False, persist="disk")
def build_flexible_logic(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not mapping.get("buyer"):
        raise ValueError("Please select a Buyer / Account column for single-sheet mode.")

    today = pd.Timestamp.today().normalize()
    buyer_col = mapping["buyer"]

    accounts = pd.DataFrame(
        {
            "buyer": df[buyer_col].astype("string").str.strip(),
            "buyer_key": clean_key(df[buyer_col]),
        }
    )

    if mapping.get("company"):
        accounts["company"] = df[mapping["company"]].astype("string").str.strip()
    else:
        accounts["company"] = accounts["buyer"]

    # Master logic
    if mapping.get("account_status"):
        df[mapping["account_status"]] = normalize_status(df[mapping["account_status"]])
        accounts["account_status"] = df[mapping["account_status"]]
    else:
        accounts["account_status"] = "Unknown"

    if mapping.get("user_state"):
        accounts["user_state"] = df[mapping["user_state"]].astype("string").str.strip()
    else:
        accounts["user_state"] = "Unknown"

    if mapping.get("am_names"):
        accounts["am_names"] = df[mapping["am_names"]].astype("string").str.strip().replace("", pd.NA).fillna("Unassigned")
    else:
        accounts["am_names"] = "Unassigned"

    if mapping.get("team"):
        accounts["team"] = df[mapping["team"]].astype("string").str.strip()
    else:
        accounts["team"] = ""

    # Restore missing metric columns before aggregation
    accounts["outstanding_balance"] = to_number(df[mapping["outstanding_balance"]]) if mapping.get("outstanding_balance") else 0
    accounts["facility_size"] = to_number(df[mapping["facility_size"]]) if mapping.get("facility_size") else 0

    # Determine the best source for disbursement date
    disbursement_col = mapping.get("last_disbursed_date") or mapping.get("disbursed_date")
    if disbursement_col:
        accounts["last_disbursed_date"] = to_date(df[disbursement_col])
    else:
        accounts["last_disbursed_date"] = pd.NaT

    aggregate_spec = {
        "company": ("company", "first"),
        "account_status": ("account_status", "first"),
        "user_state": ("user_state", "first"),
        "team": ("team", "first"),
        "am_names": ("am_names", "first"),
        "outstanding_balance": ("outstanding_balance", "max"),
        "facility_size": ("facility_size", "max"),
        "last_disbursed_date": ("last_disbursed_date", "max"),
    }
    accounts = accounts.groupby(["buyer_key", "buyer"], as_index=False).agg(**aggregate_spec)

    facility_denominator = accounts["facility_size"].where(accounts["facility_size"] > 0)
    accounts["utilisation_pct"] = (accounts["outstanding_balance"].div(facility_denominator) * 100).fillna(0)
    accounts["utilisation_pct"] = accounts["utilisation_pct"].replace([np.inf, -np.inf], 0).fillna(0)
    accounts["utilisation_category"] = np.select(
        [accounts["utilisation_pct"] >= 70, accounts["utilisation_pct"] >= 50],
        ["High", "Medium"],
        default="Low",
    )
    
    # Critical: Ensure last_disbursed_date is datetime for aging calculation
    accounts["last_disbursed_date"] = pd.to_datetime(accounts["last_disbursed_date"])
    # If date is missing, we set days to a very large number (9999) so it doesn't show as "Active Today" (0)
    accounts["days_since_last_disbursed"] = (today - accounts["last_disbursed_date"]).dt.days.fillna(9999).astype(int)
    
    # Exclusive aging buckets
    accounts["alert_120_days"] = (accounts["days_since_last_disbursed"] > 120) & (accounts["days_since_last_disbursed"] <= 150)
    accounts["alert_150_days"] = (accounts["days_since_last_disbursed"] > 150) & (accounts["days_since_last_disbursed"] <= 180)
    accounts["alert_180_days"] = (accounts["days_since_last_disbursed"] > 180) & (accounts["days_since_last_disbursed"] <= 356)
    accounts["alert_356_days"] = (accounts["days_since_last_disbursed"] > 356) & (accounts["days_since_last_disbursed"] < 9999)

    view2_present = pd.DataFrame(
        columns=[
            "buyer",
            "buyer_key",
            "am_view2",
            "due_date_invoice",
            "settlement_date",
            "disbursed_date",
            "payment_total_usd",
            "total_advanced",
            "collect_amount",
        ]
    )
    repayments_dedup = pd.DataFrame(columns=["buyer_key", "buyer", "settlement_date", "deduped_repayment"])

    has_due = mapping.get("due_date_invoice") is not None
    has_settlement = mapping.get("settlement_date") is not None
    has_payment = mapping.get("payment_total_usd") is not None
    has_disbursed = mapping.get("disbursed_date") is not None
    has_advanced = mapping.get("total_advanced") is not None

    view2_l = pd.DataFrame()
    if has_due or has_settlement or has_payment or has_disbursed or has_advanced:
        view2_l = pd.DataFrame(
            {
                "buyer": df[buyer_col].astype("string").str.strip(),
                "buyer_key": clean_key(df[buyer_col]),
                "am_view2": df[mapping["am_view2"]].astype("string").str.strip() if mapping.get("am_view2") else "",
                "due_date_invoice": to_date(df[mapping["due_date_invoice"]]) if has_due else pd.NaT,
                "settlement_date": to_date(df[mapping["settlement_date"]]) if has_settlement else pd.NaT,
                "disbursed_date": to_date(df[mapping["disbursed_date"]]) if has_disbursed else pd.NaT,
                "payment_total_usd": to_number(df[mapping["payment_total_usd"]]) if has_payment else 0,
                "total_advanced": to_number(df[mapping["total_advanced"]]) if has_advanced else 0,
            }
        )
        lookback_date = (today - pd.DateOffset(months=1)).replace(day=1)
        present_month_mask = (
            (view2_l["due_date_invoice"] >= lookback_date)
            | (view2_l["settlement_date"] >= lookback_date)
        )
        view2_present = view2_l[present_month_mask].copy()
        view2_present["collect_amount"] = view2_present["settlement_date"].isna()

        if has_settlement and has_payment and not view2_present.empty:
            repayments_dedup = (
                view2_present.dropna(subset=["settlement_date"])
                .groupby(["buyer_key", "buyer", "settlement_date"], as_index=False)["payment_total_usd"]
                .sum()
                .rename(columns={"payment_total_usd": "deduped_repayment"})
            )

    # OB Trend: Past 6 months
    trend_data = []
    if not view2_l.empty and has_disbursed and has_advanced:
        for i in range(6, -1, -1):
            d = (today - pd.DateOffset(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
            advances = view2_l[view2_l["disbursed_date"] <= d]["total_advanced"].sum()
            settlements = view2_l[view2_l["settlement_date"] <= d]["payment_total_usd"].sum()
            trend_data.append({"Month": d.strftime("%b %Y"), "Outstanding Balance": max(0, advances - settlements)})
    ob_trend = pd.DataFrame(trend_data)

    repayment_by_buyer = repayments_dedup.groupby("buyer_key", as_index=False)["deduped_repayment"].sum()
    accounts = accounts.merge(repayment_by_buyer, on="buyer_key", how="left")
    accounts["deduped_repayment"] = accounts["deduped_repayment"].fillna(0)
    accounts["adjusted_outstanding"] = accounts["outstanding_balance"] - accounts["deduped_repayment"]
    facility_denominator = accounts["facility_size"].where(accounts["facility_size"] > 0)
    accounts["post_repayment_util"] = (accounts["adjusted_outstanding"].div(facility_denominator) * 100).fillna(0)
    accounts["post_repayment_util"] = accounts["post_repayment_util"].replace([np.inf, -np.inf], 0).fillna(0)
    accounts["low_utilisation_after_repayment"] = accounts["post_repayment_util"] < 50

    return accounts[
        [
            "buyer_key",
            "buyer",
            "company",
            "account_status",
            "user_state",
            "team",
            "am_names",
            "outstanding_balance",
            "facility_size",
            "last_disbursed_date",
            "days_since_last_disbursed",
            "alert_120_days",
            "alert_150_days",
            "alert_180_days",
            "alert_356_days",
            "utilisation_pct",
            "utilisation_category",
            "deduped_repayment",
            "adjusted_outstanding",
            "post_repayment_util",
            "low_utilisation_after_repayment",
        ]
    ], view2_present, repayments_dedup, ob_trend


def format_money(value: float) -> str:
    if pd.isna(value):
        return "$0"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"





DISPLAY_NAMES = {
    "buyer": "Buyer",
    "company": "Company",
    "account_status": "Account Status",
    "team": "Team",
    "am_names": "AM",
    "outstanding_balance": "Outstanding Balance",
    "facility_size": "Facility Size",
    "last_disbursed_date": "Last Disbursed Date",
    "days_since_last_disbursed": "Days Inactive",
    "utilisation_pct": "Utilisation %",
    "utilisation_category": "Utilisation Category",
    "deduped_repayment": "Deduped Repayment",
    "adjusted_outstanding": "Adjusted Outstanding",
    "post_repayment_util": "Post Repayment Util %",
    "low_utilisation_after_repayment": "Low Util After Repay",
    "due_date_invoice": "Due Date",
    "settlement_date": "Settlement Date",
    "disbursed_date": "Disbursed Date",
    "payment_total_usd": "Payment Total USD",
    "total_advanced": "Total Advanced",
    "am_view2": "AM Email"
}

def format_df(df: pd.DataFrame) -> pd.DataFrame:
    internal_cols = [
        "buyer_key", "buyer", "company", "account_status", "team", "am_names",
        "outstanding_balance", "facility_size", "last_disbursed_date", "days_since_last_disbursed",
        "alert_120_days", "alert_150_days", "alert_180_days", "alert_356_days", "collect_amount",
        "due_date_invoice", "settlement_date", "disbursed_date", "payment_total_usd", "total_advanced",
        "am_master", "am_view1", "am_view2"
    ]
    df_clean = df.drop(columns=[c for c in internal_cols if c in df.columns], errors='ignore')
    df_renamed = df_clean.rename(columns=DISPLAY_NAMES)
    
    # Enforce logical column sequence
    existing_cols = list(df_renamed.columns)
    if existing_cols:
        # Find the best candidate for the "Name" or "Buyer" column
        name_col = None
        for c in existing_cols:
            c_low = str(c).lower()
            if any(k in c_low for k in ["buyer", "company", "name", "importer", "cp"]):
                name_col = c
                break
        
        if not name_col:
            name_col = existing_cols[0]
            
        remaining_cols = [c for c in existing_cols if c != name_col]
        
        important_metrics = [
            "Utilisation %", 
            "Post Repayment Util %", 
            "Deduped Repayment",
            "Adjusted Outstanding", 
            "Utilisation Category",
            "Days Inactive"
        ]
        
        metric_cols_present = [c for c in important_metrics if c in remaining_cols]
        other_cols = [c for c in remaining_cols if c not in metric_cols_present]
        
        final_order = [name_col] + metric_cols_present + other_cols
        return df_renamed[final_order]
        
    return df_renamed

def dataframe_config() -> Dict[str, object]:
    return {
        "Facility Size": st.column_config.NumberColumn("Facility Size", format="$ %.0f"),
        "Outstanding Balance": st.column_config.NumberColumn("Outstanding Balance", format="$ %.0f"),
        "Utilisation %": st.column_config.NumberColumn("Utilisation %", format="%.1f%%"),
        "Deduped Repayment": st.column_config.NumberColumn("Deduped Repayment", format="$ %.0f"),
        "Adjusted Outstanding": st.column_config.NumberColumn("Adjusted Outstanding", format="$ %.0f"),
        "Post Repayment Util %": st.column_config.NumberColumn("Post Repayment Util %", format="%.1f%%"),
        "Payment Total USD": st.column_config.NumberColumn("Payment Total USD", format="$ %.0f"),
        "Total Advanced": st.column_config.NumberColumn("Total Advanced", format="$ %.0f"),
        "Pending Amount": st.column_config.NumberColumn("Pending Amount", format="$ %.0f"),
        "Last Disbursed Date": st.column_config.DateColumn("Last Disbursed Date"),
        "Due Date": st.column_config.DateColumn("Due Date"),
        "Settlement Date": st.column_config.DateColumn("Settlement Date"),
        "Disbursed Date": st.column_config.DateColumn("Disbursed Date"),
    }


def render_upload_help() -> None:
    st.info(
        "Upload the Masterdata file and the Invoice Data file for the full AM logic."
    )


def get_sample_data() -> LoadedData:
    today = pd.Timestamp.today().normalize()
    
    master_df = pd.DataFrame({
        "Buyer": ["Acme Corp", "Globex", "Initech", "Umbrella Corp", "Soylent Corp"],
        "Account Status": ["Workable-Active", "Workable-Inactive (AM)", "Workable-Active", "Workable-Temporarily suspended", "Workable-Active"],
        "AM": ["Alice Smith", "Bob Jones", "Alice Smith", "Charlie Brown", "Alice Smith"],
        "Team": ["Direct", "Direct", "Indirect", "Direct", "Direct"],
        "Facility Size": [1000000, 500000, 250000, 750000, 1200000],
        "OB": [850000, 450000, 50000, 700000, 100000],
        "Last Disbursed Date": [
            today - pd.Timedelta(days=10),
            today - pd.Timedelta(days=200),
            today - pd.Timedelta(days=5),
            today - pd.Timedelta(days=30),
            today - pd.Timedelta(days=190)
        ]
    })
    
    view1_df = pd.DataFrame({
        "company": master_df["Buyer"],
        "AM Name": master_df["AM"],
        "Facility Size": master_df["Facility Size"],
        "Outstanding Balance": master_df["OB"],
        "Last Disbursed Date": master_df["Last Disbursed Date"]
    })
    
    # Generate historical transactions for View 2 to show trend
    tx_data = []
    buyers = ["Acme Corp", "Globex", "Initech", "Umbrella Corp", "Soylent Corp"]
    for i in range(12): # 12 months of history
        month_offset = today - pd.DateOffset(months=i)
        for buyer in buyers:
            # Advance
            tx_data.append({
                "Buyer": buyer,
                "AM Email": "am@example.com",
                "due date of invoice": month_offset + pd.Timedelta(days=30),
                "settlement date": month_offset + pd.Timedelta(days=25) if i > 0 else pd.NaT,
                "disbursement date": month_offset,
                "payment total usd": 100000,
                "total advanced": 100000
            })
    
    view2_df = pd.DataFrame(tx_data)
    
    return LoadedData(
        master=master_df,
        view1=view1_df,
        view2=view2_df,
        flexible=None,
        source_mode="Sample Data",
        mode="full"
    )


def get_new_selection(event, chart_key):
    if "prev_sel" not in st.session_state:
        st.session_state.prev_sel = {}
    current_points = []
    if event and hasattr(event, "selection"):
        current_points = getattr(event.selection, "points", [])
    prev_points = st.session_state.prev_sel.get(chart_key, [])
    if current_points != prev_points:
        st.session_state.prev_sel[chart_key] = current_points
        if current_points:
            return current_points[0]
    return None

@st.dialog("Drilldown Details", width="large")
def show_details(df_subset: pd.DataFrame, title: str):
    st.markdown(f"### {title}")
    st.caption(f"Showing {len(df_subset)} records.")
    st.dataframe(
        format_df(df_subset),
        use_container_width=True,
        hide_index=True,
        column_config=dataframe_config()
    )


def main() -> None:
    inject_style()

    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False

    with st.sidebar:
        st.header("Data Upload")
        if st.button("Load Sample Data", use_container_width=True):
            st.session_state["use_sample"] = True

        master_file = st.file_uploader("Masterdata file", type=["xlsx", "xls"], key="master_file")
        invoice_file = st.file_uploader("Invoice Data file", type=["xlsx", "xls"], key="invoice_file")

        if master_file or invoice_file:
            st.session_state["use_sample"] = False
        


    st.title("AM Portfolio Dashboard")

    if st.session_state["use_sample"]:
        loaded = get_sample_data()
    else:
        loaded = load_uploaded_data(master_file, invoice_file)

    if loaded is None:
        render_upload_help()
        return

    try:
        accounts, view2_present, repayments_dedup, ob_trend, view2_full = build_logic(loaded.master, loaded.view1, loaded.view2)
    except Exception as exc:
        st.error(str(exc))
        return
    orig_master_cols = list(loaded.master.columns)
    orig_view2_cols = list(loaded.view2.columns)
    account_filter_cols = [c for c in accounts.columns if c in orig_master_cols]
    view2_filter_cols = [c for c in view2_full.columns if c in orig_view2_cols and c not in account_filter_cols]

    with st.sidebar:
        st.divider()
        st.header("Custom Filters")
        st.markdown("**Masterdata**")
        selected_account_filters = st.multiselect("Add filter", account_filter_cols, default=[], key="ms_master")
        account_filter_values = {}
        if selected_account_filters:
            for col_name in selected_account_filters:
                options = sorted(accounts[col_name].dropna().astype(str).unique().tolist())
                account_filter_values[col_name] = st.multiselect(col_name, options, default=[])

        st.markdown("**Invoice Data**")
        selected_view2_filters = st.multiselect("Add filter", view2_filter_cols, default=[], key="ms_invoice")
        view2_filter_values = {}
        if selected_view2_filters:
            for col_name in selected_view2_filters:
                options = sorted(view2_full[col_name].dropna().astype(str).unique().tolist())
                view2_filter_values[col_name] = st.multiselect(col_name, options, default=[])

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    with f_col1:
        am_options = sorted(accounts["am_names"].dropna().astype(str).unique().tolist())
        selected_am = st.multiselect("AM Filter", am_options, default=[], key="main_am")
    
    with f_col2:
        if "settlement_date" in view2_full.columns:
            settlement_dates = view2_full["settlement_date"].dropna().dt.strftime("%Y-%m-%d").unique().tolist()
            settlement_options = sorted(settlement_dates)
        else:
            settlement_options = []
        selected_settlement = st.multiselect("Settlement Filter", settlement_options, default=[], key="main_settlement")

    with f_col3:
        type_col = next((c for c in accounts.columns if "type" in str(c).lower()), None)
        if not type_col:
            type_col = next((c for c in accounts.columns if "team" in str(c).lower()), "team")
        type_options = sorted(accounts[type_col].dropna().astype(str).unique().tolist()) if type_col in accounts.columns else []
        selected_type = st.multiselect("Type Filter", type_options, default=[], key="main_type")
        
    with f_col4:
        status_col = None
        for c in ["Account_Status", "Account Status", "ACCOUNT_STATUS", "USER_UTILIZATION_STATUS"]:
            if c in accounts.columns:
                status_col = c
                break
        if not status_col:
            status_col = "account_status"
        
        status_options = [
            "Non Workable",
            "Workable - Active",
            "Workable - InActive BD",
            "Workable - InActive AM",
            "Workable - Temp Suspended"
        ]
        selected_status = st.multiselect("Account Status Filter", status_options, default=[], key="main_status")

    st.divider()

    # Layout placeholders for Top Sections
    metrics_placeholder = st.container()
    tabs_placeholder = st.container()

    filtered_accounts = accounts.copy()
    
    # Apply Main Dashboard Filters
    if selected_am:
        filtered_accounts = filtered_accounts[filtered_accounts["am_names"].astype(str).isin(selected_am)]
    if selected_status:
        filtered_accounts = filtered_accounts[filtered_accounts[status_col].astype(str).isin(selected_status)]
    if selected_type and type_col in filtered_accounts.columns:
        filtered_accounts = filtered_accounts[filtered_accounts[type_col].astype(str).isin(selected_type)]
        
    # Apply Masterdata custom filters
    for col_name, selected_vals in account_filter_values.items():
        if selected_vals:
            filtered_accounts = filtered_accounts[filtered_accounts[col_name].astype(str).isin(selected_vals)]

    # Apply Invoice custom filters and Main Settlement filter
    if any(view2_filter_values.values()) or selected_settlement:
        for col_name, selected_vals in view2_filter_values.items():
            if selected_vals:
                view2_full = view2_full[view2_full[col_name].astype(str).isin(selected_vals)]
                
        if selected_settlement and "settlement_date" in view2_full.columns:
            view2_full = view2_full[view2_full["settlement_date"].dt.strftime("%Y-%m-%d").isin(selected_settlement)]
            
        valid_buyers = view2_full["buyer_key"].unique()
        filtered_accounts = filtered_accounts[filtered_accounts["buyer_key"].isin(valid_buyers)]

    # Sync all sub-datasets to the final filtered_accounts
    final_buyers = filtered_accounts["buyer_key"].unique()
    view2_full = view2_full[view2_full["buyer_key"].isin(final_buyers)]
    view2_present = view2_present[view2_present["buyer_key"].isin(final_buyers)]
    repayments_dedup = repayments_dedup[repayments_dedup["buyer_key"].isin(final_buyers)]

    # Recalculate OB Trend reactively based on filtered buyers
    filtered_buyer_keys = set(filtered_accounts["buyer_key"].unique())
    today = pd.Timestamp.today().normalize()
    
    reactive_trend_data = []
    if not view2_full.empty:
        # Filter ledger to only include buyers currently in view
        filtered_ledger = view2_full[view2_full["buyer_key"].isin(filtered_buyer_keys)]
        total_facility = filtered_accounts["facility_size"].sum()
        
        for i in range(6, -1, -1):
            d = (today - pd.DateOffset(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
            advances = filtered_ledger[filtered_ledger["disbursed_date"] <= d]["total_advanced"].sum()
            settlements = filtered_ledger[filtered_ledger["settlement_date"] <= d]["payment_total_usd"].sum()
            ob = max(0, advances - settlements)
            util = (ob / total_facility * 100) if total_facility > 0 else 0
            
            reactive_trend_data.append({
                "Month": d.strftime("%b %Y"),
                "Outstanding Balance": ob,
                "Facility": total_facility,
                "Util %": util,
                "Label": f"{format_money(ob)} ({util:.1f}%)"
            })
    
    ob_trend_reactive = pd.DataFrame(reactive_trend_data)
    with metrics_placeholder:
        # Calculate Current Month Utilisation from view2_present (Current Month mask)
        this_month_start = pd.Timestamp.today().normalize().replace(day=1)
        this_month_view2 = view2_present[view2_present["disbursed_date"] >= this_month_start]
        
        current_month_util = 0.0
        if not this_month_view2.empty and "total_advanced" in this_month_view2.columns:
            total_limit = filtered_accounts["facility_size"].sum()
            total_advanced = this_month_view2["total_advanced"].sum()
            if total_limit > 0:
                current_month_util = (total_advanced / total_limit) * 100

        metric_cols = st.columns(6)
        metric_cols[0].metric("Accounts", f"{len(filtered_accounts):,}")
        metric_cols[1].metric("Facility Size", format_money(filtered_accounts["facility_size"].sum()))
        metric_cols[2].metric("Outstanding Balance", format_money(filtered_accounts["outstanding_balance"].sum()))
        metric_cols[3].metric("Avg Utilisation", f"{filtered_accounts['utilisation_pct'].mean() if len(filtered_accounts) else 0:,.1f}%")
        metric_cols[4].metric("CM Utilisation", f"{current_month_util:,.1f}%")
        metric_cols[5].metric("Unassigned", f"{(filtered_accounts['am_names'] == 'Unassigned').sum():,}")

    with tabs_placeholder:
        overview_tab, view2_tab, inactive_tab, top_tab, account_analysis_tab = st.tabs(
            [
                "Portfolio",
                "Invoices & Repayments",
                "Inactive Accounts",
                "Top Accounts",
                "Account Analysis",
            ]
        )

        with overview_tab:
            st.header("Portfolio Health")

            left, right = st.columns([1.05, 1])
            with left:
                st.subheader("Utilisation Category")
                util_order = ["High", "Medium", "Low"]
                util_counts = (
                    filtered_accounts["utilisation_category"]
                    .value_counts()
                    .reindex(util_order, fill_value=0)
                    .reset_index()
                )
                util_counts.columns = ["Utilisation Category", "Accounts"]
                fig = px.bar(
                    util_counts,
                    y="Utilisation Category",
                    x="Accounts",
                    orientation="h",
                    color="Utilisation Category",
                    text="Accounts",
                    color_discrete_map={"High": "#157f3b", "Medium": "#c47a00", "Low": "#b42318"},
                )
                fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10), height=330)
                event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="chart_util_cat")
                pt = get_new_selection(event, "chart_util_cat")
                if pt:
                    cat = pt.get("y")
                    subset = filtered_accounts[filtered_accounts["utilisation_category"] == cat]
                    show_details(subset, f"Accounts with {cat} Utilisation")

            with right:
                st.subheader("Distribution by User State")
                state_data = filtered_accounts.dropna(subset=["user_state"]).copy()
                if not state_data.empty:
                    state_data["state_abbr"] = state_data["user_state"].map(STATE_ABBR).fillna(state_data["user_state"])
                    by_state = (
                        state_data.groupby("state_abbr", as_index=False)
                        .agg(accounts=("buyer", "nunique"))
                    )
                    fig = px.choropleth(
                        by_state,
                        locations="state_abbr",
                        locationmode="USA-states",
                        color="accounts",
                        color_continuous_scale=BLUE_SCALE,
                        scope="usa",
                        labels={"accounts": "Unique Accounts", "state_abbr": "State"}
                    )
                    fig.update_layout(dragmode=False, margin=dict(l=10, r=10, t=20, b=10), height=330)
                    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="chart_user_state", config={'scrollZoom': False, 'displayModeBar': False})
                    pt = get_new_selection(event, "chart_user_state")
                    if pt:
                        state_clicked = pt.get("location", pt.get("x"))
                        subset = state_data[state_data["state_abbr"] == state_clicked]
                        show_details(subset, f"Accounts in {state_clicked}")
                else:
                    st.info("No user state data available.")

            st.subheader("Industry Distribution of Accounts")
            if "industry" in filtered_accounts.columns and not filtered_accounts["industry"].isna().all():
                by_industry = (
                    filtered_accounts.groupby("industry", as_index=False)
                    .agg(accounts=("buyer", "nunique"))
                    .sort_values("accounts", ascending=False)
                    .head(15)
                )
                fig = px.pie(
                    by_industry,
                    values="accounts",
                    names="industry",
                    custom_data=["industry"],
                    hole=0.4,
                    color_discrete_sequence=THEME_PALETTE,
                    labels={"industry": "Industry", "accounts": "Accounts"}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=True, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05), margin=dict(l=10, r=10, t=20, b=10), height=380)
                event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="chart_industry")
                pt = get_new_selection(event, "chart_industry")
                if pt:
                    ind_clicked = pt.get("customdata", [None])[0]
                    subset = filtered_accounts[filtered_accounts["industry"] == ind_clicked]
                    show_details(subset, f"Accounts in {ind_clicked} Industry")
            else:
                st.info("No industry data available.")


        with view2_tab:
            st.header("Expected Payments & Repayments")
            col1, col2 = st.columns(2)
            with col1:
                # Default from_date to 15 days ago so past repayments are visible by default
                from_date = st.date_input("From Date", pd.Timestamp.today().normalize() - pd.Timedelta(days=15))
            with col2:
                to_date = st.date_input("To Date", pd.Timestamp.today().normalize() + pd.Timedelta(days=15))

            from_dt = pd.to_datetime(from_date)
            to_dt = pd.to_datetime(to_date)

            def get_filtered_expected(df, start, end):
                if df.empty or "due_date_invoice" not in df.columns: return pd.DataFrame()
                mask = (df["due_date_invoice"] >= start) & (df["due_date_invoice"] <= end) & (df["settlement_date"].isna())
                return df[mask]
                
            def get_filtered_repayments(df, start, end):
                if df.empty or "settlement_date" not in df.columns: return pd.DataFrame()
                mask = (df["settlement_date"] >= start) & (df["settlement_date"] <= end)
                return df[mask]

            expected_filtered = get_filtered_expected(view2_present, from_dt, to_dt)
            repay_filtered = get_filtered_repayments(view2_present, from_dt, to_dt)

            e1, e2 = st.columns(2)
            e1.metric("Expected Payment Accounts", f"{expected_filtered['buyer_key'].nunique() if 'buyer_key' in expected_filtered.columns else 0:,}")
            e1.metric("Expected Payments Amount", format_money(expected_filtered['payment_total_usd'].sum() if not expected_filtered.empty else 0))
            
            e2.metric("Repayment Accounts", f"{repay_filtered['buyer_key'].nunique() if 'buyer_key' in repay_filtered.columns else 0:,}")
            e2.metric("Repayments Amount", format_money(repay_filtered['payment_total_usd'].sum() if not repay_filtered.empty else 0))

            t1, t2 = st.columns(2)
            with t1:
                st.markdown("#### Expected Payments")
                if not expected_filtered.empty:
                    exp_by_date = expected_filtered.groupby("due_date_invoice", as_index=False)["payment_total_usd"].sum()
                    fig_exp = px.bar(exp_by_date, x="due_date_invoice", y="payment_total_usd", color_discrete_sequence=["#c47a00"])
                    fig_exp.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=300)
                    event1 = st.plotly_chart(fig_exp, use_container_width=True, on_select="rerun", key="chart_exp_payments")
                    pt1 = get_new_selection(event1, "chart_exp_payments")
                    if pt1:
                        date_clicked = pt1.get("x")
                        subset = expected_filtered[expected_filtered["due_date_invoice"] == pd.to_datetime(date_clicked)]
                        show_details(subset, f"Expected Payments on {date_clicked}")
                else:
                    st.info("No expected payments in this range.")

            with t2:
                st.markdown("#### Repayments")
                if not repay_filtered.empty:
                    rep_by_date = repay_filtered.groupby("settlement_date", as_index=False)["payment_total_usd"].sum()
                    fig_rep = px.bar(rep_by_date, x="settlement_date", y="payment_total_usd", color_discrete_sequence=["#157f3b"])
                    fig_rep.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=300)
                    event2 = st.plotly_chart(fig_rep, use_container_width=True, on_select="rerun", key="chart_repayments")
                    pt2 = get_new_selection(event2, "chart_repayments")
                    if pt2:
                        date_clicked = pt2.get("x")
                        subset = repay_filtered[repay_filtered["settlement_date"] == pd.to_datetime(date_clicked)]
                        show_details(subset, f"Repayments on {date_clicked}")
                else:
                    st.info("No repayments in this range.")

            st.divider()
            
            repay_by_date = repayments_dedup.groupby("settlement_date", as_index=False)["deduped_repayment"].sum()
            if not repay_by_date.empty:
                st.subheader("Unique Account Repayment Trends")
                fig = px.line(
                    repay_by_date,
                    x="settlement_date",
                    y="deduped_repayment",
                    markers=True,
                    labels={"settlement_date": "Settlement Date", "deduped_repayment": "Deduped Repayment"},
                )
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=310)
                event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="chart_repay_trends")
                pt = get_new_selection(event, "chart_repay_trends")
                if pt:
                    date_clicked = pt.get("x")
                    subset = repayments_dedup[repayments_dedup["settlement_date"] == pd.to_datetime(date_clicked)]
                    show_details(subset, f"Repayments on {date_clicked}")

        with inactive_tab:
            st.header("Inactive Accounts")
            # Update to match new explicit category string
            inactive = filtered_accounts[filtered_accounts["account_status"] == "Workable - InActive AM"].copy()
            
            st.markdown("#### All Inactive Accounts")
            
            i_row1 = st.columns(3)
            i_row1[0].metric("Total Inactive Accounts", f"{len(inactive):,}")
            i_row1[1].metric("Total Facility Size", format_money(inactive["facility_size"].sum()))
            i_row1[2].metric("Total Outstanding Balance", format_money(inactive["outstanding_balance"].sum()))
            
            st.divider()
            if not inactive.empty:
                fig_inact = px.bar(
                    inactive.sort_values("days_since_last_disbursed", ascending=False).head(50),
                    x="buyer",
                    y="days_since_last_disbursed",
                    color="outstanding_balance",
                    color_continuous_scale=TEAL_SCALE,
                    labels={"buyer": "Account", "days_since_last_disbursed": "Days Inactive", "outstanding_balance": "Outstanding Balance"}
                )
                fig_inact.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=400)
                event = st.plotly_chart(fig_inact, use_container_width=True, on_select="rerun", key="chart_inactive")
                pt = get_new_selection(event, "chart_inactive")
                if pt:
                    buyer_clicked = pt.get("x")
                    subset = inactive[inactive["buyer"] == buyer_clicked]
                    show_details(subset, f"Inactive Account Details: {buyer_clicked}")
            else:
                st.info("No inactive accounts found.")


        with top_tab:
            st.header("Top Accounts")
            top_left, top_right = st.columns(2)

            with top_left:
                st.subheader("Account Exposure Overview")
                scatter_df = filtered_accounts.dropna(subset=["facility_size", "outstanding_balance"]).copy()
                if not scatter_df.empty:
                    fig_scatter = px.scatter(
                        scatter_df,
                        x="facility_size",
                        y="outstanding_balance",
                        color="account_status",
                        size="utilisation_pct",
                        hover_name="buyer",
                        custom_data=["buyer"],
                        labels={"facility_size": "Facility Size", "outstanding_balance": "Outstanding Balance", "account_status": "Status", "utilisation_pct": "Utilisation %"},
                        size_max=30,
                        opacity=0.7,
                    )
                    fig_scatter.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=430)
                    event = st.plotly_chart(fig_scatter, use_container_width=True, on_select="rerun", key="chart_exposure")
                    pt = get_new_selection(event, "chart_exposure")
                    if pt:
                        buyer_clicked = pt.get("customdata", [None])[0]
                        subset = filtered_accounts[filtered_accounts["buyer"] == buyer_clicked]
                        show_details(subset, f"Exposure Details: {buyer_clicked}")
                else:
                    st.info("No facility size or outstanding balance data available.")

            with top_right:
                st.subheader("Top 50 by Facility Size")
                top50 = filtered_accounts.sort_values("facility_size", ascending=False).head(50)
                fig = px.bar(
                    top50.head(20).sort_values("facility_size"),
                    x="facility_size",
                    y="buyer",
                    orientation="h",
                    color="account_status",
                    labels={"facility_size": "Facility Size", "buyer": "Buyer", "account_status": "Status"},
                )
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=430)
                event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="chart_top50")
                pt = get_new_selection(event, "chart_top50")
                if pt:
                    buyer_clicked = pt.get("y")
                    subset = filtered_accounts[filtered_accounts["buyer"] == buyer_clicked]
                    show_details(subset, f"Facility Details: {buyer_clicked}")

        with account_analysis_tab:
            st.header("Account Analysis")
            if filtered_accounts.empty:
                st.info("No accounts available for the selected AM(s).")
            else:
                account_options = filtered_accounts["buyer_key"].unique()
                buyer_map = dict(zip(filtered_accounts["buyer_key"], filtered_accounts["company"]))
                selected_buyer_key = st.selectbox(
                    "Select an Account",
                    options=account_options,
                    format_func=lambda x: f"{buyer_map.get(x, x)} ({x})"
                )

                if selected_buyer_key:
                    st.divider()
                    st.subheader(f"Analysis for {buyer_map.get(selected_buyer_key, selected_buyer_key)}")
                    
                    acc_view2 = view2_full[view2_full["buyer_key"] == selected_buyer_key].copy()
                    
                    if acc_view2.empty:
                        st.warning("No invoice data found for this account.")
                    else:
                        c1, c2 = st.columns(2)
                        
                        unique_exporters = acc_view2["exporter_company"].nunique() if "exporter_company" in acc_view2.columns and not acc_view2["exporter_company"].isna().all() else 0
                        unique_invoices = acc_view2["invoice_id"].nunique() if "invoice_id" in acc_view2.columns and not acc_view2["invoice_id"].isna().all() else len(acc_view2)
                        
                        c1.metric("Unique Exporters", f"{unique_exporters:,}")
                        c2.metric("Unique Invoices Funded", f"{unique_invoices:,}")
                        
                        st.markdown("#### Repayment & Margin Track")
                        repayment_counts = {}
                        settled_invoices = acc_view2.dropna(subset=["settlement_date", "due_date_invoice"]).copy()
                        if not settled_invoices.empty:
                            def classify_repayment(row):
                                diff = (row["settlement_date"] - row["due_date_invoice"]).days
                                if diff < -3: return "Prior Payment"
                                elif diff <= 3: return "On Time"
                                else: return "Delayed"
                            settled_invoices["repayment_status"] = settled_invoices.apply(classify_repayment, axis=1)
                            repayment_counts = settled_invoices["repayment_status"].value_counts().to_dict()
                        
                        margin_counts = {}
                        if "margin_received_date" in acc_view2.columns:
                            margin_paid = acc_view2.dropna(subset=["margin_received_date", "disbursed_date"]).copy()
                            if not margin_paid.empty:
                                def classify_margin(row):
                                    diff = (row["margin_received_date"] - row["disbursed_date"]).days
                                    if diff < -3: return "Prior Payment"
                                    elif diff <= 3: return "On Time"
                                    else: return "Delayed"
                                margin_paid["margin_status"] = margin_paid.apply(classify_margin, axis=1)
                                margin_counts = margin_paid["margin_status"].value_counts().to_dict()
                        
                        if repayment_counts or margin_counts:
                            track_data = []
                            for status in ["Prior Payment", "On Time", "Delayed"]:
                                track_data.append({"Track": "Repayment", "Status": status, "Count": repayment_counts.get(status, 0)})
                                track_data.append({"Track": "Margin", "Status": status, "Count": margin_counts.get(status, 0)})
                            track_df = pd.DataFrame(track_data)
                            
                            track_df["Total"] = track_df.groupby("Track")["Count"].transform("sum")
                            track_df["Percentage"] = (track_df["Count"] / track_df["Total"]) * 100
                            track_df["Percentage"] = track_df["Percentage"].fillna(0)
                            
                            fig_track = px.bar(
                                track_df,
                                y="Track",
                                x="Percentage",
                                color="Status",
                                orientation="h",
                                barmode="stack",
                                color_discrete_map={"Prior Payment": "#157f3b", "On Time": "#3b82f6", "Delayed": "#b42318"}
                            )
                            fig_track.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=250, xaxis_title="% of Invoices", yaxis_title="")
                            event = st.plotly_chart(fig_track, use_container_width=True, on_select="rerun", key="chart_track")
                            pt = get_new_selection(event, "chart_track")
                            if pt:
                                track_clicked = pt.get("y")
                                show_details(acc_view2, f"Invoice History: {selected_buyer_key} ({track_clicked} track)")
                        else:
                            st.info("No repayment or margin data available to track.")
                            
                        col_trend, col_geo = st.columns(2)
                        with col_trend:
                            st.markdown("#### Invoice Submission Trend")
                            trend_df = acc_view2.dropna(subset=["disbursed_date"]).copy()
                            if not trend_df.empty:
                                trend_df["disbursed_month"] = trend_df["disbursed_date"].dt.to_period("M").dt.to_timestamp()
                                trend_grouped = trend_df.groupby("disbursed_month").size().reset_index(name="Invoice Count")
                                fig_trend = px.line(
                                    trend_grouped, 
                                    x="disbursed_month", 
                                    y="Invoice Count", 
                                    markers=True,
                                    color_discrete_sequence=["#0d9488"],
                                    labels={"disbursed_month": "Disbursement Month", "Invoice Count": "Invoices Submitted"}
                                )
                                fig_trend.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=300)
                                event = st.plotly_chart(fig_trend, use_container_width=True, on_select="rerun", key="chart_trend")
                                pt = get_new_selection(event, "chart_trend")
                                if pt:
                                    month_clicked = pt.get("x")
                                    show_details(acc_view2, f"Invoices around {month_clicked}")
                            else:
                                st.info("No disbursed dates available for trend analysis.")
                                
                        with col_geo:
                            if "exporter_country" in acc_view2.columns and not acc_view2["exporter_country"].isna().all():
                                st.markdown("#### Exporter Country Analysis")
                                country_counts = acc_view2["exporter_country"].value_counts().reset_index()
                                country_counts.columns = ["Country", "Count"]
                                fig_country = px.scatter_geo(
                                    country_counts,
                                    locations="Country",
                                    locationmode="country names",
                                    size="Count",
                                    color="Count",
                                    color_continuous_scale=TEAL_SCALE,
                                    projection="orthographic",
                                )
                                fig_country.update_geos(
                                    showcountries=True, countrycolor="#cbd5e1",
                                    showocean=True, oceancolor="#f8fafc",
                                    showland=True, landcolor="#e2e8f0",
                                    showframe=False,
                                    projection_rotation={"lat": 20, "lon": -40}
                                )
                                fig_country.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                                event = st.plotly_chart(fig_country, use_container_width=True, on_select="rerun", key="chart_globe")
                                pt = get_new_selection(event, "chart_globe")
                                if pt:
                                    country_clicked = pt.get("location")
                                    subset = acc_view2[acc_view2["exporter_country"] == country_clicked]
                                    show_details(subset, f"Invoices for Exporter Country: {country_clicked}")
                            else:
                                st.info("No exporter country data available.")


if __name__ == "__main__":
    main()
