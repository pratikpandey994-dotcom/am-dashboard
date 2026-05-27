# Smart Analytics Dashboard - Data & Visualization Design

## 1. Data Sources Available

The dashboard processes two primary types of data exports, representing different grains of information:

### A. Account-Level Data (Portfolio View)
**Granularity:** One row per Account (Buyer/Company).
**Key Data Points:**
- **Identifiers & Ownership:** `company`, `AM_Name`, `AM_Email`
- **Limits & Utilization:** `Facility_Size` (Limit), `Outstanding_Balance` (Utilization), `max_balance`, `overdraft_limit`
- **Pricing & Terms:** `Signed-up IRR`, `interest_rate`, `flat_discounting_fee`, `overdue_rate`
- **Risk & Status:** `utilization_status`, `suspension_reason`, `manual_suspension_reason`, `latest_suspension_Date`
- **Performance:** `Total Orginations`, `Realised Revenue`

### B. Transaction-Level Data (Invoice View)
**Granularity:** One row per Invoice/Transaction.
**Key Data Points:**
- **Identifiers:** `Invoice ID`, `Buyer`, `Seller`, `AM_Email`
- **Invoice Financials:** `Outstanding`, `Total Advanced`, `booked_revenue`, `total_fees`, `advice_collection_amount`
- **Timelines & Risk:** `Stage` (closed/active), `Payment_terms`, `disbursed_date`, `due_date_of_invoice`, `settlement_date`, `dpd` (Days Past Due)

---

## 2. Proposed Data Mapping Strategy

To ensure the dashboard works dynamically regardless of which file is uploaded, the auto-mapping logic should identify the following standard business terms:

| Standard Concept | Potential Column Matches in Data | Purpose |
| :--- | :--- | :--- |
| **Account / Buyer** | `company`, `buyer`, `client`, `customer` | Grouping data by client. |
| **Account Manager** | `am_name`, `am_email`, `owner` | Filtering the dashboard for a specific AM. |
| **Facility Limit** | `facility_size`, `limit`, `approved_limit` | Baseline for how much an account can borrow. |
| **Utilization (Outstanding)**| `outstanding_balance`, `outstanding` | How much of the limit is currently used. |
| **Revenue / Fees** | `realised revenue`, `booked_revenue`, `total_fees`| Tracking profitability. |
| **Risk / Status** | `utilization_status`, `stage`, `suspension_reason`| Identifying accounts that are active, suspended, or closed. |
| **Days Past Due (DPD)** | `dpd`, `overdue_days` | Identifying late payments (Transaction view). |

---

## 3. Visualization & Dashboard Layout

The dashboard should be divided into sections focusing on different insights for Account Managers.

### A. High-Level KPIs (Top Row)
- **Total Facility Limit:** Sum of all limits across active accounts.
- **Total Outstanding (Utilization):** Sum of all outstanding balances.
- **Overall Utilization %:** (Total Outstanding / Total Limit) * 100.
- **Accounts at Risk:** Count of accounts suspended OR invoices with DPD > 0.

### B. Limits vs. Utilization Analysis
- **Visualization:** Grouped Bar Chart or Bullet Chart.
- **X-Axis:** Top Accounts (by Outstanding or Limit).
- **Y-Axis:** Dual bars showing `Facility Limit` and `Outstanding Balance` side-by-side.
- **Insight:** Quickly identify which accounts are maxed out (near 100% utilization) and which have headroom for more volume.

### C. Risk & Health Monitoring
- **Visualization (Risk Distribution):** Donut Chart.
- **Data:** Group by `utilization_status` (e.g., Active vs. Suspended) OR group by `DPD` buckets (e.g., Current, 1-15 days late, 15-30 days late, 30+ days late).
- **Insight:** Helps AMs prioritize follow-ups for collections or understand why accounts are blocked.

### D. Revenue & Yield (Optional, based on data)
- **Visualization:** Scatter Plot or Bubble Chart.
- **X-Axis:** Total Originations / Total Advanced.
- **Y-Axis:** Realized Revenue.
- **Color:** IRR / Interest Rate.
- **Insight:** Identify high-volume, highly profitable accounts vs. low-margin accounts.

### E. Smart Filter / Drill-Down
- Allow filtering the entire dashboard by **Account Manager** (using `AM_Name` or `AM_Email`).
- When a specific account is clicked or selected from a dropdown, show a raw data table of their specific metrics and any suspension reasons.
