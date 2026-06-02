from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


KEEP_STATUSES = {
    "Workable-Active",
    "Workable-Inactive (AM)",
    "Workable-Temporarily suspended",
}


@dataclass(frozen=True)
class LoadedData:
    master: Optional[pd.DataFrame]
    view1: Optional[pd.DataFrame]
    view2: Optional[pd.DataFrame]
    flexible: Optional[pd.DataFrame]
    source_mode: str
    mode: str


st.set_page_config(page_title="AM Portfolio Dashboard", layout="wide")


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1540px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        div[data-testid="stMetricLabel"] {
            color: #475467;
        }
        div[data-testid="stMetricValue"] {
            color: #101828;
        }
        .section-note {
            color: #667085;
            font-size: 0.92rem;
            margin-top: -0.6rem;
            margin-bottom: 0.8rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_name(value: object) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def normalize_status(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+-\s+", "-", regex=True)
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


def load_uploaded_data(single_file, master_file, view1_file, view2_file) -> Optional[LoadedData]:
    if single_file is not None:
        sheets = workbook_sheets(single_file)
        classified: Dict[str, pd.DataFrame] = {}
        for sheet_name, df in sheets.items():
            role = classify_sheet(sheet_name, df)
            if role and role not in classified:
                classified[role] = df

        if {"master", "view1", "view2"}.issubset(classified):
            return LoadedData(
                master=classified["master"],
                view1=classified["view1"],
                view2=classified["view2"],
                flexible=None,
                source_mode=f"Single workbook: {single_file.name}",
                mode="full",
            )

        if len(sheets) == 1:
            sheet_name, df = next(iter(sheets.items()))
            return LoadedData(
                master=None,
                view1=None,
                view2=None,
                flexible=df,
                source_mode=f"Single sheet: {single_file.name} / {sheet_name}",
                mode="flexible",
            )

        if classified:
            preferred_role = next((role for role in ("master", "view1", "view2") if role in classified), None)
            return LoadedData(
                master=classified.get("master"),
                view1=classified.get("view1"),
                view2=classified.get("view2"),
                flexible=classified[preferred_role] if preferred_role else next(iter(sheets.values())),
                source_mode=f"Partial workbook: {single_file.name}",
                mode="flexible",
            )

        return LoadedData(
            master=None,
            view1=None,
            view2=None,
            flexible=next(iter(sheets.values())),
            source_mode=f"Workbook mapped as single sheet: {single_file.name}",
            mode="flexible",
        )

    if master_file and view1_file and view2_file:
        return LoadedData(
            master=pd.read_excel(master_file),
            view1=pd.read_excel(view1_file),
            view2=pd.read_excel(view2_file),
            flexible=None,
            source_mode="Three uploaded files",
            mode="full",
        )

    partial_files = [
        ("Masterdata", master_file),
        ("View 1 - Company", view1_file),
        ("View 2 - Invoices / Repayments", view2_file),
    ]
    uploaded_partial = [(label, file) for label, file in partial_files if file is not None]
    if len(uploaded_partial) == 1:
        label, file = uploaded_partial[0]
        return LoadedData(
            master=None,
            view1=None,
            view2=None,
            flexible=pd.read_excel(file),
            source_mode=f"Single uploaded file: {label} / {file.name}",
            mode="flexible",
        )

    return None


@st.cache_data(show_spinner=False)
def build_logic(master: pd.DataFrame, view1: pd.DataFrame, view2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master_cols = {
        "Buyer": first_existing(master, ["Buyer"]),
        "Account_Status": first_existing(master, ["Account_Status", "Account Status"]),
        "AM": first_existing(master, ["AM"]),
        "Team": first_existing(master, ["Team"]),
        "Facility_Size": first_existing(master, ["Facility_Size", "Facility Size"]),
        "OB": first_existing(master, ["OB", "Outstanding Balance"]),
        "Last_Disbursed_Date": first_existing(master, ["Last_Disbursed_Date", "Last Disbursed Date"]),
    }
    view1_cols = {
        "company": first_existing(view1, ["company", "Buyer"]),
        "AM_Name": first_existing(view1, ["AM_Name", "AM Name", "AM"]),
        "Facility_Size": first_existing(view1, ["Facility_Size", "Facility Size"]),
        "Outstanding_Balance": first_existing(view1, ["Outstanding_Balance", "Outstanding Balance"]),
        "Last_Disbursed_Date": first_existing(view1, ["Last_Disbursed_Date", "Last Disbursed Date"]),
    }
    view2_cols = {
        "Buyer": first_existing(view2, ["Buyer"]),
        "AM_Email": first_existing(view2, ["AM_Email", "AM Email"]),
        "due_date_of_invoice": first_existing(view2, ["due_date_of_invoice", "due date of invoice"]),
        "settlement_date": first_existing(view2, ["settlement_date", "settlement date"]),
        "payment_total_usd": first_existing(view2, ["payment_total_usd", "payment total usd"]),
    }
    require_columns(master, master_cols, "Masterdata")
    require_columns(view1, view1_cols, "View 1")
    require_columns(view2, view2_cols, "View 2")

    master_l = master.assign(
        buyer=master[master_cols["Buyer"]].astype("string").str.strip(),
        buyer_key=clean_key(master[master_cols["Buyer"]]),
        account_status=normalize_status(master[master_cols["Account_Status"]]),
        am_master=master[master_cols["AM"]].astype("string").str.strip(),
        team=master[master_cols["Team"]].astype("string").str.strip(),
        facility_size_master=to_number(master[master_cols["Facility_Size"]]),
        outstanding_balance_master=to_number(master[master_cols["OB"]]),
        last_disbursed_date_master=to_date(master[master_cols["Last_Disbursed_Date"]]),
    )
    master_l = master_l[master_l["account_status"].isin(KEEP_STATUSES)].copy()

    view1_l = view1.assign(
        buyer_key=clean_key(view1[view1_cols["company"]]),
        company=view1[view1_cols["company"]].astype("string").str.strip(),
        am_view1=view1[view1_cols["AM_Name"]].astype("string").str.strip(),
        facility_size_view1=to_number(view1[view1_cols["Facility_Size"]]),
        outstanding_balance_view1=to_number(view1[view1_cols["Outstanding_Balance"]]),
        last_disbursed_date_view1=to_date(view1[view1_cols["Last_Disbursed_Date"]]),
    )[[
        "buyer_key",
        "company",
        "am_view1",
        "facility_size_view1",
        "outstanding_balance_view1",
        "last_disbursed_date_view1",
    ]]

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
    accounts["days_since_last_disbursed"] = (today - accounts["last_disbursed_date"]).dt.days
    accounts["alert_180_days"] = accounts["days_since_last_disbursed"] > 180

    view2_l = view2.assign(
        buyer=view2[view2_cols["Buyer"]].astype("string").str.strip(),
        buyer_key=clean_key(view2[view2_cols["Buyer"]]),
        am_view2=view2[view2_cols["AM_Email"]].astype("string").str.strip(),
        due_date_invoice=to_date(view2[view2_cols["due_date_of_invoice"]]),
        settlement_date=to_date(view2[view2_cols["settlement_date"]]),
        payment_total_usd=to_number(view2[view2_cols["payment_total_usd"]]),
    )[["buyer", "buyer_key", "am_view2", "due_date_invoice", "settlement_date", "payment_total_usd"]]

    present_month = (
        ((view2_l["due_date_invoice"].dt.month == today.month) & (view2_l["due_date_invoice"].dt.year == today.year))
        | ((view2_l["settlement_date"].dt.month == today.month) & (view2_l["settlement_date"].dt.year == today.year))
    )
    view2_present = view2_l[present_month].copy()
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

    keep_account_cols = [
        "buyer",
        "company",
        "account_status",
        "team",
        "am_names",
        "outstanding_balance",
        "facility_size",
        "last_disbursed_date",
        "days_since_last_disbursed",
        "alert_180_days",
        "utilisation_pct",
        "utilisation_category",
        "deduped_repayment",
        "adjusted_outstanding",
        "post_repayment_util",
        "low_utilisation_after_repayment",
    ]
    return accounts[keep_account_cols], view2_present, repayments_dedup


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
        buyer = column_selector("Buyer / Account", df, ["Buyer", "company", "Account", "Company"], required=True)
    with row1[1]:
        account_status = column_selector("Account Status", df, ["Account_Status", "Account Status", "Status", "Stage"])
    with row1[2]:
        am_names = column_selector("AM Name", df, ["AM", "AM_Name", "AM Name", "Account Manager", "Owner"])
    with row1[3]:
        team = column_selector("Team", df, ["Team"])

    row2 = st.columns(4)
    with row2[0]:
        company = column_selector("Company", df, ["company", "Company", "Buyer"])
    with row2[1]:
        outstanding_balance = column_selector(
            "Outstanding Balance",
            df,
            ["Outstanding_Balance", "Outstanding Balance", "OB", "Outstanding"],
        )
    with row2[2]:
        facility_size = column_selector("Facility Size", df, ["Facility_Size", "Facility Size", "Limit"])
    with row2[3]:
        last_disbursed_date = column_selector(
            "Last Disbursed Date",
            df,
            ["Last_Disbursed_Date", "Last Disbursed Date", "disbursed_date", "Disbursed Date"],
        )

    row3 = st.columns(4)
    with row3[0]:
        due_date_invoice = column_selector(
            "Due Date of Invoice",
            df,
            ["due_date_of_invoice", "Due Date of Invoice", "due date", "Due Date"],
        )
    with row3[1]:
        settlement_date = column_selector("Settlement Date", df, ["settlement_date", "Settlement Date"])
    with row3[2]:
        payment_total_usd = column_selector(
            "Payment Total USD",
            df,
            ["payment_total_usd", "Payment Total USD", "payment", "Amount"],
        )
    with row3[3]:
        am_view2 = column_selector("View 2 AM / Email", df, ["AM_Email", "AM Email", "AM"])

    return {
        "buyer": buyer,
        "account_status": account_status,
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
    }


@st.cache_data(show_spinner=False)
def build_flexible_logic(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    if mapping.get("account_status"):
        accounts["account_status"] = normalize_status(df[mapping["account_status"]])
        if accounts["account_status"].isin(KEEP_STATUSES).any():
            accounts = accounts[accounts["account_status"].isin(KEEP_STATUSES)].copy()
    else:
        accounts["account_status"] = "Unknown"

    if mapping.get("am_names"):
        accounts["am_names"] = df[mapping["am_names"]].astype("string").str.strip().replace("", pd.NA).fillna("Unassigned")
    else:
        accounts["am_names"] = "Unassigned"

    if mapping.get("team"):
        accounts["team"] = df[mapping["team"]].astype("string").str.strip()
    else:
        accounts["team"] = ""

    accounts["outstanding_balance"] = to_number(df[mapping["outstanding_balance"]]) if mapping.get("outstanding_balance") else 0
    accounts["facility_size"] = to_number(df[mapping["facility_size"]]) if mapping.get("facility_size") else 0
    accounts["last_disbursed_date"] = to_date(df[mapping["last_disbursed_date"]]) if mapping.get("last_disbursed_date") else pd.NaT

    aggregate_spec = {
        "company": ("company", "first"),
        "account_status": ("account_status", "first"),
        "team": ("team", "first"),
        "am_names": ("am_names", "first"),
        "outstanding_balance": ("outstanding_balance", "sum"),
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
    accounts["days_since_last_disbursed"] = (today - accounts["last_disbursed_date"]).dt.days
    accounts["alert_180_days"] = accounts["days_since_last_disbursed"] > 180

    view2_present = pd.DataFrame(columns=["buyer", "buyer_key", "am_view2", "due_date_invoice", "settlement_date", "payment_total_usd", "collect_amount"])
    repayments_dedup = pd.DataFrame(columns=["buyer_key", "buyer", "settlement_date", "deduped_repayment"])

    has_due = mapping.get("due_date_invoice") is not None
    has_settlement = mapping.get("settlement_date") is not None
    has_payment = mapping.get("payment_total_usd") is not None
    if has_due or has_settlement or has_payment:
        view2_l = pd.DataFrame(
            {
                "buyer": df[buyer_col].astype("string").str.strip(),
                "buyer_key": clean_key(df[buyer_col]),
                "am_view2": df[mapping["am_view2"]].astype("string").str.strip() if mapping.get("am_view2") else "",
                "due_date_invoice": to_date(df[mapping["due_date_invoice"]]) if has_due else pd.NaT,
                "settlement_date": to_date(df[mapping["settlement_date"]]) if has_settlement else pd.NaT,
                "payment_total_usd": to_number(df[mapping["payment_total_usd"]]) if has_payment else 0,
            }
        )
        present_month = (
            ((view2_l["due_date_invoice"].dt.month == today.month) & (view2_l["due_date_invoice"].dt.year == today.year))
            | ((view2_l["settlement_date"].dt.month == today.month) & (view2_l["settlement_date"].dt.year == today.year))
        )
        view2_present = view2_l[present_month].copy()
        view2_present["collect_amount"] = view2_present["settlement_date"].isna()

        if has_settlement and has_payment and not view2_present.empty:
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

    return accounts[
        [
            "buyer",
            "company",
            "account_status",
            "team",
            "am_names",
            "outstanding_balance",
            "facility_size",
            "last_disbursed_date",
            "days_since_last_disbursed",
            "alert_180_days",
            "utilisation_pct",
            "utilisation_category",
            "deduped_repayment",
            "adjusted_outstanding",
            "post_repayment_util",
            "low_utilisation_after_repayment",
        ]
    ], view2_present, repayments_dedup


def format_money(value: float) -> str:
    if pd.isna(value):
        return "$0"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def filter_accounts(df: pd.DataFrame, selected_ams: list[str], selected_statuses: list[str]) -> pd.DataFrame:
    filtered = df.copy()
    if selected_ams:
        filtered = filtered[filtered["am_names"].isin(selected_ams)]
    if selected_statuses:
        filtered = filtered[filtered["account_status"].isin(selected_statuses)]
    return filtered


def dataframe_config() -> Dict[str, object]:
    return {
        "facility_size": st.column_config.NumberColumn("Facility Size", format="$ %.0f"),
        "outstanding_balance": st.column_config.NumberColumn("Outstanding", format="$ %.0f"),
        "utilisation_pct": st.column_config.NumberColumn("Utilisation %", format="%.1f%%"),
        "deduped_repayment": st.column_config.NumberColumn("Deduped Repayment", format="$ %.0f"),
        "adjusted_outstanding": st.column_config.NumberColumn("Adjusted Outstanding", format="$ %.0f"),
        "post_repayment_util": st.column_config.NumberColumn("Post Repayment Util %", format="%.1f%%"),
        "payment_total_usd": st.column_config.NumberColumn("Payment Total USD", format="$ %.0f"),
        "deduped_repayment": st.column_config.NumberColumn("Deduped Repayment", format="$ %.0f"),
        "last_disbursed_date": st.column_config.DateColumn("Last Disbursed"),
        "due_date_invoice": st.column_config.DateColumn("Due Date"),
        "settlement_date": st.column_config.DateColumn("Settlement Date"),
    }


def render_upload_help() -> None:
    st.info(
        "Upload one sheet for flexible mapping, one workbook with all three sheets, or three separate files for the full AM logic."
    )


def main() -> None:
    inject_style()

    st.title("AM Portfolio Dashboard")
    st.caption("Portfolio utilisation, 180-day alerts, present-month invoices, repayments, and Team Direct coverage.")

    with st.expander("Data Upload", expanded=True):
        upload_cols = st.columns(4)
        with upload_cols[0]:
            single_file = st.file_uploader(
                "Workbook or single-sheet file",
                type=["xlsx", "xls"],
                key="single_file",
            )
        with upload_cols[1]:
            master_file = st.file_uploader("Masterdata file", type=["xlsx", "xls"], key="master_file")
        with upload_cols[2]:
            view1_file = st.file_uploader("View 1 - Company file", type=["xlsx", "xls"], key="view1_file")
        with upload_cols[3]:
            view2_file = st.file_uploader("View 2 - Invoices / Repayments file", type=["xlsx", "xls"], key="view2_file")

    loaded = load_uploaded_data(single_file, master_file, view1_file, view2_file)
    if loaded is None:
        render_upload_help()
        return

    st.success(loaded.source_mode)

    try:
        if loaded.mode == "full":
            accounts, view2_present, repayments_dedup = build_logic(loaded.master, loaded.view1, loaded.view2)
        else:
            with st.expander("Column Mapping", expanded=True):
                st.info("Single-sheet mode: select the columns to use for dashboard calculations.")
                mapping = render_flexible_mapping(loaded.flexible)
                accounts, view2_present, repayments_dedup = build_flexible_logic(loaded.flexible, mapping)
    except Exception as exc:
        st.error(str(exc))
        return

    with st.expander("Filters", expanded=True):
        am_options = sorted(accounts["am_names"].dropna().astype(str).unique().tolist())
        status_options = sorted(accounts["account_status"].dropna().astype(str).unique().tolist())
        filter_cols = st.columns(2)
        with filter_cols[0]:
            selected_ams = st.multiselect("AM", am_options, default=[])
        with filter_cols[1]:
            selected_statuses = st.multiselect("Account Status", status_options, default=status_options)

    filtered_accounts = filter_accounts(accounts, selected_ams, selected_statuses)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Accounts", f"{len(filtered_accounts):,}")
    metric_cols[1].metric("Facility Size", format_money(filtered_accounts["facility_size"].sum()))
    metric_cols[2].metric("Outstanding", format_money(filtered_accounts["outstanding_balance"].sum()))
    metric_cols[3].metric("Avg Utilisation", f"{filtered_accounts['utilisation_pct'].mean() if len(filtered_accounts) else 0:,.1f}%")
    metric_cols[4].metric("180-Day Alerts", f"{filtered_accounts['alert_180_days'].sum():,}")
    metric_cols[5].metric("Unassigned", f"{(filtered_accounts['am_names'] == 'Unassigned').sum():,}")

    st.divider()

    overview_tab, view2_tab, inactive_tab, top_tab, data_tab = st.tabs(
        [
            "Portfolio",
            "Invoices & Repayments",
            "Inactive AM",
            "Top Accounts",
            "Data",
        ]
    )

    with overview_tab:
        st.header("Portfolio Health")
        st.markdown(
            '<div class="section-note">Company-level view using Masterdata status, View 1 balances, and 180-day last-disbursed alerts.</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.05, 1])
        with left:
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
                x="Utilisation Category",
                y="Accounts",
                color="Utilisation Category",
                text="Accounts",
                color_discrete_map={"High": "#157f3b", "Medium": "#c47a00", "Low": "#b42318"},
            )
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10), height=330)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            by_am = (
                filtered_accounts.groupby("am_names", as_index=False)
                .agg(
                    accounts=("buyer", "count"),
                    facility_size=("facility_size", "sum"),
                    outstanding_balance=("outstanding_balance", "sum"),
                    alerts=("alert_180_days", "sum"),
                )
                .sort_values("outstanding_balance", ascending=False)
                .head(15)
            )
            fig = px.bar(
                by_am,
                y="am_names",
                x="outstanding_balance",
                orientation="h",
                color="alerts",
                color_continuous_scale=["#d1fadf", "#f04438"],
                labels={"am_names": "AM", "outstanding_balance": "Outstanding", "alerts": "180-Day Alerts"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=330, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Accounts Requiring Attention")
        attention = filtered_accounts[
            filtered_accounts["alert_180_days"] | filtered_accounts["low_utilisation_after_repayment"]
        ].sort_values(["alert_180_days", "post_repayment_util"], ascending=[False, True])
        st.dataframe(
            attention[
                [
                    "buyer",
                    "am_names",
                    "account_status",
                    "facility_size",
                    "outstanding_balance",
                    "utilisation_pct",
                    "last_disbursed_date",
                    "alert_180_days",
                    "post_repayment_util",
                    "low_utilisation_after_repayment",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config=dataframe_config(),
        )

    with view2_tab:
        st.header("Present-Month Invoices & Repayments")
        st.markdown(
            '<div class="section-note">Rows where invoice due date or settlement date falls in the current calendar month.</div>',
            unsafe_allow_html=True,
        )

        v2_left, v2_mid, v2_right = st.columns(3)
        v2_left.metric("Present-Month Rows", f"{len(view2_present):,}")
        v2_mid.metric("Collect Amount Rows", f"{view2_present['collect_amount'].sum():,}")
        v2_right.metric("Deduped Repayment", format_money(repayments_dedup["deduped_repayment"].sum()))

        repay_by_date = repayments_dedup.groupby("settlement_date", as_index=False)["deduped_repayment"].sum()
        if not repay_by_date.empty:
            fig = px.line(
                repay_by_date,
                x="settlement_date",
                y="deduped_repayment",
                markers=True,
                labels={"settlement_date": "Settlement Date", "deduped_repayment": "Deduped Repayment"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=310)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Present-Month View 2 Rows")
        st.dataframe(
            view2_present.sort_values(["collect_amount", "due_date_invoice"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
            column_config=dataframe_config(),
        )

        st.subheader("Deduped Repayments by Buyer and Settlement Date")
        st.dataframe(
            repayments_dedup.sort_values("deduped_repayment", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config=dataframe_config(),
        )

    with inactive_tab:
        st.header("Workable-Inactive AM")
        inactive = filtered_accounts[filtered_accounts["account_status"] == "Workable-Inactive (AM)"].copy()
        i1, i2, i3 = st.columns(3)
        i1.metric("Inactive AM Accounts", f"{len(inactive):,}")
        i2.metric("Facility Size", format_money(inactive["facility_size"].sum()))
        i3.metric("180-Day Alerts", f"{inactive['alert_180_days'].sum():,}")
        st.dataframe(
            inactive.sort_values("facility_size", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config=dataframe_config(),
        )

    with top_tab:
        st.header("Top Accounts")
        top_left, top_right = st.columns(2)

        with top_left:
            st.subheader("Top 15 by Facility Size")
            top15 = filtered_accounts.sort_values("facility_size", ascending=False).head(15)
            fig = px.bar(
                top15.sort_values("facility_size"),
                x="facility_size",
                y="buyer",
                orientation="h",
                color="utilisation_pct",
                color_continuous_scale=["#216e8c", "#f79009", "#b42318"],
                labels={"facility_size": "Facility Size", "buyer": "Buyer", "utilisation_pct": "Utilisation %"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=430)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top15, use_container_width=True, hide_index=True, column_config=dataframe_config())

        with top_right:
            st.subheader("Top 50 Team Direct by Facility Size")
            team_direct = filtered_accounts[filtered_accounts["team"].astype("string").str.casefold() == "direct"]
            top50 = team_direct.sort_values("facility_size", ascending=False).head(50)
            fig = px.bar(
                top50.head(20).sort_values("facility_size"),
                x="facility_size",
                y="buyer",
                orientation="h",
                color="account_status",
                labels={"facility_size": "Facility Size", "buyer": "Buyer", "account_status": "Status"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=430)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top50, use_container_width=True, hide_index=True, column_config=dataframe_config())

    with data_tab:
        st.header("Logical Data")
        st.subheader("Accounts Logic Table")
        st.dataframe(filtered_accounts, use_container_width=True, hide_index=True, column_config=dataframe_config())
        st.subheader("Raw Present-Month View 2 Logic Table")
        st.dataframe(view2_present, use_container_width=True, hide_index=True, column_config=dataframe_config())


if __name__ == "__main__":
    main()
