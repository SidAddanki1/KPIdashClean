# C-Store KPI Insights Suite — UX-Polished + Fixed Filters
# - Global sidebar: Department only (compact)
# - Local per-tab filters with isolated keys and "All = no filter"
# - Options sourced from raw_df (full domain)
# - Defaults start empty (no pre-selections)
# - KPI, Trends (+ conservative forecast), Top-N, Basket Affinity, Price Ladder, Store Map,
#   Assortment & Space Optimization (Productivity, Rationalization, Opportunity Map, New Items)
# - Visual improvements: card layout, consistent spacing, legends, progressive disclosure,
#   unobtrusive help text, metric definitions everywhere, reduced visual noise.

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from collections import Counter
from itertools import combinations
from typing import Dict, Tuple, List
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------------------------------------------------------
# Page & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="C-Store KPI Insights Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)
pio.templates.default = "plotly_white"

# Soft CSS polish (spacing, cards, headers)
st.markdown("""
<style>
/* Tighten top padding */
.block-container { padding-top: 1.25rem; }
/* Headline */
h1, h2, h3 { letter-spacing: 0.2px; }
/* Card feel */
div[data-testid="stMetric"] { border:1px solid rgba(250,250,250,0.08); border-radius:12px; padding:8px; }
section[data-testid="stSidebar"] .block-container { padding-top: 0.5rem; }
.small-note { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
.help-note { color: rgba(255,255,255,0.75); }
hr { margin: 0.5rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def to_num_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

@st.cache_data(show_spinner=False)
def load_and_normalize(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Canonicalize common headers
    rename_map = {
        "Brand": "Brand_Name",
        "Units_Sold": "Quantity",
        "Total_Sales": "Total_Sale",
        "Payment_Type": "Payment_Method",
        "StoreID": "Store_ID",
        "TransactionID": "Transaction_ID",
        "UnitPrice": "Unit_Price",
        "Product": "Item",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    need = {"Date", "Transaction_ID", "Store_ID", "Category", "Item"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Transaction_ID"])
    for c in ["Transaction_ID", "Store_ID", "Item", "Brand_Name", "Category", "Payment_Method"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    if "Unit_Price" in df.columns: df["Unit_Price"] = to_num_clean(df["Unit_Price"]).fillna(0.0)
    if "Quantity" in df.columns:   df["Quantity"] = to_num_clean(df["Quantity"]).fillna(0.0)
    if "Total_Sale" in df.columns: df["Total_Sale"] = to_num_clean(df["Total_Sale"]).fillna(0.0)

    if {"Quantity", "Unit_Price"} <= set(df.columns):
        if "Total_Sale" not in df.columns:
            df["Total_Sale"] = (df["Quantity"] * df["Unit_Price"]).round(2)
        else:
            needs = (df["Total_Sale"].isna()) | (df["Total_Sale"] == 0)
            df.loc[needs, "Total_Sale"] = (df.loc[needs, "Quantity"] * df.loc[needs, "Unit_Price"]).round(2)
    return df

# ---- Data path (no uploader per request)
DATA_PATH = "cstorereal.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"CSV not found at {os.path.abspath(DATA_PATH)}. Please add the file and redeploy.")
    st.stop()

raw_df = load_and_normalize(DATA_PATH)
if raw_df.empty:
    st.stop()

# Full-domain option lists (ALWAYS from raw_df)
DOMAIN = {
    "stores":   sorted(raw_df["Store_ID"].unique().tolist()),
    "cats":     sorted(raw_df["Category"].unique().tolist()),
    "brands":   sorted(raw_df["Brand_Name"].unique().tolist()) if "Brand_Name" in raw_df.columns else [],
    "prods":    sorted(raw_df["Item"].unique().tolist()),
    "pays":     sorted(raw_df["Payment_Method"].unique().tolist()) if "Payment_Method" in raw_df.columns else [],
    "min_date": raw_df["Date"].min().date(),
    "max_date": raw_df["Date"].max().date(),
}

# -----------------------------------------------------------------------------
# Global: Department (compact)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.caption("Filter")
    st.markdown("### Department")
    DEPARTMENT_TAB_MAP = {
        "Executive / Strategy & Finance": ["📊 KPI Overview", "📈 KPI Trends"],
        "Merchandising / Category": ["🏆 Top-N", "📦 Assortment & Space"],
        "Marketing & CRM": ["🧺 Basket Affinity", "📈 KPI Trends"],
        "Operations": ["🗺️ Store Map", "📊 KPI Overview"],
        "Price & Promo": ["💲 Price Ladder", "📈 KPI Trends"],
        "All": ["📊 KPI Overview","📈 KPI Trends","🏆 Top-N","🧺 Basket Affinity","💲 Price Ladder","🗺️ Store Map","📦 Assortment & Space"],
    }
    dept = st.selectbox(
        "Department",
        list(DEPARTMENT_TAB_MAP.keys()),
        index=list(DEPARTMENT_TAB_MAP.keys()).index("All"),
        help="Controls which reports (tabs) are visible. Local filters live inside each tab."
    )

# -----------------------------------------------------------------------------
# Shared helpers (filters, KPIs, trends, forecasts, affinity)
# -----------------------------------------------------------------------------
RULE_MAP = {"Daily":"D", "Weekly":"W-SUN", "Monthly":"MS"}

def multiselect_all_none(label, options, key, help_text=None):
    """Multiselect with 'All' option that behaves as NO FILTER when chosen. Starts empty."""
    opts = ["All"] + list(options)
    sel = st.multiselect(label, opts, default=[], key=key, help=help_text)
    if "All" in sel or len(sel) == 0:
        return None  # no filter
    return [v for v in sel if v != "All"]

def apply_local_filters(df, stores, cats, brands, prods, pays, start_date, end_date):
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    out = df.loc[mask].copy()
    if stores: out = out[out["Store_ID"].isin(stores)]
    if cats:   out = out[out["Category"].isin(cats)]
    if brands and "Brand_Name" in out.columns: out = out[out["Brand_Name"].isin(brands)]
    if prods:  out = out[out["Item"].isin(prods)]
    if pays and "Payment_Method" in out.columns: out = out[out["Payment_Method"].isin(pays)]
    return out

@st.cache_data(show_spinner=False)
def kpis(df: pd.DataFrame) -> Dict[str, float]:
    ts = df["Total_Sale"].sum() if "Total_Sale" in df.columns else 0.0
    tq = df["Quantity"].sum()    if "Quantity"   in df.columns else 0.0
    tx = df["Transaction_ID"].nunique() if "Transaction_ID" in df.columns else 0
    spb = (ts/tx) if tx else 0.0
    asp = (ts/tq) if tq else 0.0
    return dict(total_sales=ts, total_qty=tq, tx=tx, spend_per_basket=spb, asp=asp)

@st.cache_data(show_spinner=False)
def trends(df: pd.DataFrame, rule: str, group_dim: str|None) -> pd.DataFrame:
    gcols = [pd.Grouper(key="Date", freq=rule)]
    if group_dim and group_dim in df.columns:
        gcols.append(group_dim)
    t = (df.groupby(gcols, dropna=False)
           .agg(Total_Sale=("Total_Sale","sum"),
                Quantity=("Quantity","sum"),
                Transactions=("Transaction_ID","nunique"))
           .reset_index()
           .sort_values("Date"))
    t["Spend per Basket"] = np.where(t["Transactions"]>0, t["Total_Sale"]/t["Transactions"], 0.0)
    return t

def _forecast_freq_meta(rule:str):
    if rule == "D": return 30, 7, "D", "next 30 days"
    if rule.startswith("W-"): return 4, 52, rule, "next 4 weeks"
    if rule == "MS": return 1, 12, "MS", "next month"
    return 30, 7, "D", "next 30 days"

def _series(df, rule, metric):
    s = (df.set_index("Date")[metric].sort_index().resample(rule).sum().astype(float).fillna(0.0))
    s.index.name = "Date"; return s

def _fit_ets_safe(ts, rule, seasonal_periods):
    use_fixed = rule.startswith("W-")
    use_season = seasonal_periods and (len(ts) >= 2*seasonal_periods)
    try:
        trend_type = 'add' if use_fixed else None
        fit_params = {'smoothing_level':0.1,'smoothing_trend':0.01,'optimized':False} if use_fixed else {'optimized':True}
        model = ExponentialSmoothing(
            ts, trend=trend_type, damped_trend=False,
            seasonal=("add" if use_season else None),
            seasonal_periods=(seasonal_periods if use_season else None),
            initialization_method="estimated"
        ).fit(**fit_params)
        fitted = model.fittedvalues.reindex(ts.index)
        resid_std = float(np.nanstd((ts - fitted).to_numpy()))
        return model, resid_std
    except Exception:
        return None, None

def _seasonal_naive(ts, steps, sp):
    if sp and len(ts) >= sp:
        rep = np.tile(ts.iloc[-sp:].to_numpy(), int(np.ceil(steps/sp)))[:steps]
        return rep.astype(float)
    return np.full(steps, float(ts.iloc[-1]) if len(ts) else 0.0)

def _choose_model(ts, rule, sp):
    n = len(ts)
    if n < max(12, sp+4): return "naive", None, None
    h = max(6, int(round(n*0.12)))
    train, hold = ts.iloc[:-h], ts.iloc[-h:]
    ets_model, resid_std = _fit_ets_safe(train, rule, sp)
    naive_fc = _seasonal_naive(train, h, sp)
    ets_fc = ets_model.forecast(h).to_numpy() if ets_model else None
    def smape(a,f):
        a = a.astype(float); f = f.astype(float)
        d = (np.abs(a)+np.abs(f)); d[d==0]=1.0
        return float(np.mean(2.0*np.abs(a-f)/d))
    a = hold.to_numpy()
    if ets_fc is None or smape(a, naive_fc) <= smape(a, ets_fc):
        return "naive", None, None
    return "ets", ets_model, resid_std

@st.cache_data(show_spinner=False)
def forecast(df, rule, metric, alpha=0.05):
    steps, sp, freq, label = _forecast_freq_meta(rule)
    ts = _series(df, rule, metric).clip(lower=0)
    mtype, model, resid_std = _choose_model(ts, rule, sp)
    if mtype=="ets" and model is not None:
        mean = model.forecast(steps).to_numpy()
        z = 1.96 if alpha==0.05 else 1.64
        band = (resid_std or 0.0)*z
        lo = np.maximum(0.0, mean-band); hi = np.maximum(0.0, mean+band)
    else:
        mean = _seasonal_naive(ts, steps, sp)
        lo = np.maximum(0.0, mean*0.9); hi = mean*1.1
    look = min(len(ts), 28 if rule=="D" else (12 if rule=="MS" else 12))
    recent = float(ts.iloc[-look:].mean()) if look>0 else 0.0
    cap_lo, cap_hi = recent*steps*0.7, recent*steps*1.3
    s = float(mean.sum())
    if cap_hi>0 and s>cap_hi:
        scale = cap_hi/s; mean*=scale; lo*=scale; hi*=scale
    elif s<cap_lo and s>0:
        scale = cap_lo/s; mean*=scale; lo*=scale; hi*=scale
    idx = pd.date_range(ts.index[-1], periods=steps+1, freq=freq)[1:]
    return pd.DataFrame({"Date":idx, "yhat":mean, "yhat_lower":lo, "yhat_upper":hi}), label

@st.cache_data(show_spinner=False)
def affinity_rules(df, key_col):
    scope = df.dropna(subset=["Transaction_ID", key_col]).copy()
    scope["Transaction_ID"] = scope["Transaction_ID"].astype(str)
    scope[key_col] = scope[key_col].astype(str)
    tx_total = scope["Transaction_ID"].nunique()
    if tx_total==0: return pd.DataFrame()

    basket_sales = scope.groupby("Transaction_ID")["Total_Sale"].sum()
    tx_keys = scope.groupby("Transaction_ID")[key_col].apply(lambda s: tuple(sorted(set(s))))
    item_counts, pair_counts = Counter(), Counter()
    for keys in tx_keys:
        for k in keys: item_counts[k]+=1
        for a,b in combinations(keys,2): pair_counts[tuple(sorted((a,b)))] += 1
    if not pair_counts: return pd.DataFrame()
    item_txids = scope.groupby(key_col)["Transaction_ID"].apply(set)

    def qty_in(item, tx_ids):
        if "Quantity" not in scope.columns or not tx_ids: return 0.0
        v = scope[(scope["Transaction_ID"].isin(tx_ids)) & (scope[key_col]==item)]
        return float(v["Quantity"].sum())

    rows=[]
    for (a,b), n_ab in pair_counts.items():
        ca, cb = item_counts[a], item_counts[b]
        sup = n_ab/tx_total
        lift = sup/((ca/tx_total)*(cb/tx_total)) if (ca and cb) else 0.0
        tx_a, tx_b = item_txids.get(a,set()), item_txids.get(b,set())
        co = tx_a & tx_b
        co_sales = float(basket_sales.loc[list(co)].sum()) if co else 0.0
        co_avg = (co_sales/n_ab) if n_ab else 0.0
        qa, qb = qty_in(a, co), qty_in(b, co)
        rows += [
            {"Antecedent":a,"Consequent":b,"Total Co-Baskets":n_ab,"Support (A,B)":sup,"Confidence (A->B)":(n_ab/ca) if ca else 0.0,"Lift (A,B)":lift,
             "Total_Antecedent_Qty_in_CoBasket":qa,"Avg_Antecedent_Qty_in_CoBasket":(qa/n_ab) if n_ab else 0.0,
             "Total_CoBasket_Sales_Value":co_sales,"Avg_CoBasket_Spend":co_avg},
            {"Antecedent":b,"Consequent":a,"Total Co-Baskets":n_ab,"Support (A,B)":sup,"Confidence (A->B)":(n_ab/cb) if cb else 0.0,"Lift (A,B)":lift,
             "Total_Antecedent_Qty_in_CoBasket":qb,"Avg_Antecedent_Qty_in_CoBasket":(qb/n_ab) if n_ab else 0.0,
             "Total_CoBasket_Sales_Value":co_sales,"Avg_CoBasket_Spend":co_avg}
        ]
    out = pd.DataFrame(rows).sort_values(["Lift (A,B)","Confidence (A->B)"], ascending=[False,False]).reset_index(drop=True)
    return out

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("## C-Store KPI Insights Suite")
st.caption("Clean layout • Local filters • Helpful tooltips • Consistent definitions • Progressive disclosure")

# -----------------------------------------------------------------------------
# Tab registry (shown based on department)
# -----------------------------------------------------------------------------
ALL_TABS = {
    "📊 KPI Overview": {},
    "📈 KPI Trends": {},
    "🏆 Top-N": {},
    "🧺 Basket Affinity": {},
    "💲 Price Ladder": {},
    "🗺️ Store Map": {},
    "📦 Assortment & Space": {},
}
VISIBLE_TABS = [t for t in ALL_TABS if t in DEPARTMENT_TAB_MAP.get(dept, [])]

# Create tabs
tabs = st.tabs(VISIBLE_TABS)

# -----------------------------------------------------------------------------
# Local filter block (helper)
# -----------------------------------------------------------------------------
def local_filter_block(prefix: str):
    colA, colB, colC, colD, colE, colF = st.columns([1.2,1.2,1.2,1.8,1.2,1.2])
    with colA:
        stores = multiselect_all_none("Store(s)", DOMAIN["stores"], key=f"{prefix}_stores",
                                      help_text="Pick specific stores or leave empty/choose All to include all.")
    with colB:
        cats = multiselect_all_none("Category", DOMAIN["cats"], key=f"{prefix}_cats")
    with colC:
        brands = multiselect_all_none("Brand", DOMAIN["brands"], key=f"{prefix}_brands") if DOMAIN["brands"] else None
    with colD:
        prods = multiselect_all_none("Product", DOMAIN["prods"], key=f"{prefix}_prods")
    with colE:
        pays = multiselect_all_none("Payment Method", DOMAIN["pays"], key=f"{prefix}_pays") if DOMAIN["pays"] else None
    with colF:
        freq = st.radio("Time", ["Daily","Weekly","Monthly"], index=1, horizontal=True, key=f"{prefix}_freq")
    date_col1, date_col2 = st.columns([1,1])
    with date_col1:
        sd = st.date_input("Start", value=DOMAIN["min_date"], key=f"{prefix}_start")
    with date_col2:
        ed = st.date_input("End", value=DOMAIN["max_date"], key=f"{prefix}_end")
    return stores, cats, brands, prods, pays, RULE_MAP[freq], sd, ed

# -----------------------------------------------------------------------------
# Tab 1: KPI Overview
# -----------------------------------------------------------------------------
if "📊 KPI Overview" in VISIBLE_TABS:
    with tabs[VISIBLE_TABS.index("📊 KPI Overview")]:
        st.markdown("### KPIs")
        stores, cats, brands, prods, pays, rule, sd, ed = local_filter_block("kpi")
        df_ = apply_local_filters(raw_df, stores, cats, brands, prods, pays, sd, ed)
        if df_.empty:
            st.info("No data for the selected filters.")
        else:
            m = kpis(df_)
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Total Sales", f"${m['total_sales']:,.0f}")
            c2.metric("Transactions", f"{m['tx']:,}")
            c3.metric("Units", f"{int(m['total_qty']):,}")
            c4.metric("Spend/Basket", f"${m['spend_per_basket']:,.2f}")
            c5.metric("ASP", f"${m['asp']:,.2f}")
            with st.expander("Metric definitions", expanded=False):
                st.markdown("""
- **Total Sales**: Sum of `Total_Sale`.  
- **Transactions**: Count of unique `Transaction_ID`.  
- **Units**: Sum of `Quantity`.  
- **Spend/Basket**: Total Sales ÷ Transactions.  
- **ASP**: Total Sales ÷ Units.
                """)

# -----------------------------------------------------------------------------
# Tab 2: KPI Trends (+ Forecast)
# -----------------------------------------------------------------------------
if "📈 KPI Trends" in VISIBLE_TABS:
    with tabs[VISIBLE_TABS.index("📈 KPI Trends")]:
        st.markdown("### KPI Trends")
        stores, cats, brands, prods, pays, rule, sd, ed = local_filter_block("tr")
        df_ = apply_local_filters(raw_df, stores, cats, brands, prods, pays, sd, ed)
        if df_.empty:
            st.info("No data for the selected filters.")
        else:
            metric = st.selectbox("Metric", ["Total_Sale","Quantity","Spend per Basket","Transactions"], key="tr_metric")
            # If multiple values selected in any dimension, split by first applicable one
            group_dim = None
            choices = {
                "Store_ID": stores,
                "Category": cats,
                "Brand_Name": brands,
                "Item": prods
            }
            for d,v in choices.items():
                if v and len(v)>1: group_dim=d; break

            tdf = trends(df_, rule, group_dim)
            if tdf.empty or metric not in tdf.columns:
                st.info("No trend data for this selection.")
            else:
                fig = px.line(tdf, x="Date", y=metric, color=group_dim,
                              title=f"{metric.replace('_',' ')} over time"+(f" by {group_dim}" if group_dim else ""))
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                if (group_dim is None) and metric in ["Total_Sale","Quantity"]:
                    fc_df, horizon = forecast(df_, rule, metric)
                    fig2 = px.line(fc_df, x="Date", y="yhat", title=f"Forecast — {metric} ({horizon})")
                    fig2.add_traces([
                        dict(
                            x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
                            y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
                            fill="toself", fillcolor="rgba(99,110,250,0.15)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip", name="95% interval"
                        )
                    ])
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption(
                        f"Projected total {metric.replace('_',' ').lower()} for {horizon}: "
                        f"**{fc_df['yhat'].sum():,.0f}** "
                        f"(95% CI: {fc_df['yhat_lower'].sum():,.0f} – {fc_df['yhat_upper'].sum():,.0f})"
                    )
        with st.expander("Metric definitions", expanded=False):
            st.markdown("""
- **Spend per Basket**: Total Sales ÷ Transactions.  
- **Transactions**: Unique baskets.  
- **Forecast**: Conservative ETS vs seasonal-naive model, with capped horizon totals to avoid runaway effects.
            """)

# -----------------------------------------------------------------------------
# Tab 3: Top-N
# -----------------------------------------------------------------------------
if "🏆 Top-N" in VISIBLE_TABS:
    with tabs[VISIBLE_TABS.index("🏆 Top-N")]:
        st.markdown("### Top-N")
        stores, cats, brands, prods, pays, rule, sd, ed = local_filter_block("tn")
        df_ = apply_local_filters(raw_df, stores, cats, brands, prods, pays, sd, ed)
        if df_.empty:
            st.info("No data for the selected filters.")
        else:
            dims = [d for d in ["Category","Brand_Name","Store_ID","Item"] if d in df_.columns]
            dim = st.selectbox("Rank by dimension", dims, index=0, key="tn_dim")
            N = st.slider("N", 5, 30, 10, key="tn_n")
            top = (df_.groupby(dim)["Total_Sale"].sum().sort_values(ascending=False).head(N).reset_index())
            fig = px.bar(top, x=dim, y="Total_Sale", text_auto=".2s", title=f"Top {N} {dim} by Sales")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Metric definitions", expanded=False):
            st.markdown("- **Total Sales**: Sum of `Total_Sale` within current filters.")

# -----------------------------------------------------------------------------
# Tab 4: Basket Affinity
# -----------------------------------------------------------------------------
if "🧺 Basket Affinity" in VISIBLE_TABS:
    with tabs[VISIBLE_TABS.index("🧺 Basket Affinity")]:
        st.markdown("### Targeted Basket Affinity")
        stores, cats, brands, prods, pays, rule, sd, ed = local_filter_block("ba")
        df_ = apply_local_filters(raw_df, stores, cats, brands, prods, pays, sd, ed)
        if df_.empty:
            st.info("No data for the selected filters.")
        else:
            levels = [c for c in ["Item","Brand_Name","Category"] if c in df_.columns]
            key_col = st.radio("Granularity", levels, index=0, horizontal=True, key="ba_gr")
            target = st.selectbox("Target", sorted(df_[key_col].unique().tolist()), key="ba_target")
            rules = affinity_rules(df_, key_col)
            if rules.empty:
                st.info("No co-basket pairs found. Try widening filters.")
            else:
                t = rules[rules["Antecedent"]==str(target)].copy()
                if t.empty:
                    st.info("No associations for the selected target.")
                else:
                    t["Associated Item"] = t["Consequent"]
                    disp = t[["Associated Item","Confidence (A->B)","Lift (A,B)","Total Co-Baskets",
                              "Total_CoBasket_Sales_Value","Avg_CoBasket_Spend",
                              "Total_Antecedent_Qty_in_CoBasket","Avg_Antecedent_Qty_in_CoBasket"]].copy()
                    disp["Confidence (%)"] = (disp["Confidence (A->B)"]*100).round(2)
                    disp = disp.drop(columns=["Confidence (A->B)"]).rename(columns={"Lift (A,B)":"Lift",
                        "Total_CoBasket_Sales_Value":"Total Co-Basket Sales",
                        "Avg_CoBasket_Spend":"Avg Co-Basket Spend",
                        "Total_Antecedent_Qty_in_CoBasket":f"Total Qty of {target}",
                        "Avg_Antecedent_Qty_in_CoBasket":f"Avg Qty of {target}"
                    })
                    disp = disp.sort_values(["Lift","Confidence (%)"], ascending=False).reset_index(drop=True)
                    st.dataframe(
                        disp,
                        hide_index=True, use_container_width=True,
                        column_config={
                            "Associated Item": st.column_config.TextColumn("Associated"),
                            "Confidence (%)": st.column_config.ProgressColumn("Confidence", min_value=0,max_value=100, format="%.1f%%",
                                help="Share of target baskets that also included the associated item."),
                            "Lift": st.column_config.NumberColumn("Lift", format="%.2f", help="Co-occurrence vs independence (>1 = positive)"),
                            "Total Co-Baskets": st.column_config.NumberColumn("Co-Baskets", format="%d",
                                help="Number of baskets where both target and associated item appeared."),
                            "Total Co-Basket Sales": st.column_config.NumberColumn("Total Basket Sales", format="$%.0f"),
                            "Avg Co-Basket Spend": st.column_config.NumberColumn("Avg Basket Spend", format="$%.2f"),
                            f"Total Qty of {target}": st.column_config.NumberColumn("Total Target Qty", format="%d"),
                            f"Avg Qty of {target}": st.column_config.NumberColumn("Avg Target Qty", format="%.2f"),
                        }
                    )
        with st.expander("Metric definitions", expanded=False):
            st.markdown("""
- **Confidence**: P(B|A).  
- **Lift**: Confidence ÷ baseline(B).  
- **Co-Baskets**: Count of baskets containing both items.
            """)

# -----------------------------------------------------------------------------
# Tab 5: Price Ladder
# -----------------------------------------------------------------------------
if "💲 Price Ladder" in VISIBLE_TABS:
    with tabs[VISIBLE_TABS.index("💲 Price Ladder")]:
        st.markdown("### Price Ladder")
        stores, cats, brands, prods, pays, rule, sd, ed = local_filter_block("pl")
        df_ = apply_local_filters(raw_df, stores, cats, brands, prods, pays, sd, ed)
        if df_.empty or "Unit_Price" not in df_.columns:
            st.info("No price data in the current selection.")
        else:
            level = st.selectbox("Level", [l for l in ["Item","Brand_Name","Category"] if l in df_.columns], key="pl_level")
            agg = (df_.groupby(level).agg(Avg_Price=("Unit_Price","mean"),
                                          Median_Price=("Unit_Price","median"),
                                          Count=("Unit_Price","size")).reset_index())
            sort_by = st.selectbox("Sort by", ["Median Price","Average Price","Count"], key="pl_sort")
            sort_col = {"Median Price":"Median_Price","Average Price":"Avg_Price","Count":"Count"}[sort_by]
            agg = agg.sort_values(sort_col, ascending=False)
            fig = px.bar(agg, x=level, y="Median_Price", text_auto=".2f",
                         hover_data={"Avg_Price":":.2f","Median_Price":":.2f","Count":":,"},
                         title=f"Median price by {level}")
            fig.update_layout(xaxis_title=level, yaxis_title="Median Price")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Metric definitions", expanded=False):
            st.markdown("""
- **Median Price**: Median of Unit_Price observations.  
- **Average Price**: Mean of Unit_Price.  
- **Count**: Number of rows contributing to the calculation.
            """)

# -----------------------------------------------------------------------------
# Tab 6: Store Map
# -----------------------------------------------------------------------------
if "🗺️ Store Map" in VISIBLE_TABS:
    with tabs[VISIBLE_TABS.index("🗺️ Store Map")]:
        st.markdown("### Store Map")
        stores, cats, brands, prods, pays, rule, sd, ed = local_filter_block("sm")
        df_ = apply_local_filters(raw_df, stores, cats, brands, prods, pays, sd, ed)
        need = {"Store_ID","Store_Latitude","Store_Longitude"}
        if not need.issubset(df_.columns):
            st.info("Location columns not found (need Store_ID, Store_Latitude, Store_Longitude).")
        elif df_.empty:
            st.info("No data for the selected filters.")
        else:
            kpi = (df_.groupby("Store_ID", as_index=False)
                     .agg(Total_Sale=("Total_Sale","sum"),
                          Quantity=("Quantity","sum"),
                          Transactions=("Transaction_ID","nunique")))
            kpi["Spend_per_Basket"] = np.where(kpi["Transactions"]>0, kpi["Total_Sale"]/kpi["Transactions"], 0.0)
            kpi["ASP"] = np.where(kpi["Quantity"]>0, kpi["Total_Sale"]/kpi["Quantity"], 0.0)
            loc = df_[["Store_ID","Store_City","Store_State","Store_Latitude","Store_Longitude"]].drop_duplicates("Store_ID")
            m = kpi.merge(loc, on="Store_ID", how="left")
            m["Store_Label"] = m["Store_ID"].astype(str) + " — " + m.get("Store_City","").astype(str) + ", " + m.get("Store_State","").astype(str)

            col1, col2 = st.columns(2)
            with col1:
                size_metric = st.selectbox("Bubble Size", ["Total_Sale","Transactions","Quantity"], key="sm_size")
            with col2:
                color_metric = st.selectbox("Bubble Color", ["Total_Sale","Spend_per_Basket","ASP","Transactions","Quantity"], key="sm_color")

            fig = px.scatter_mapbox(
                m, lat="Store_Latitude", lon="Store_Longitude",
                size=size_metric, color=color_metric, size_max=28,
                zoom=3, center={"lat":39.5,"lon":-98.35}, mapbox_style="open-street-map",
                hover_name="Store_Label",
                custom_data=np.stack([
                    m["Total_Sale"].values, m["Transactions"].values, m["Quantity"].values,
                    m["Spend_per_Basket"].values, m["ASP"].values
                ], axis=-1)
            )
            fig.update_traces(hovertemplate=
                "<b>%{hovertext}</b><br>"
                "Total Sales: $%{customdata[0]:,.0f}<br>"
                "Transactions: %{customdata[1]:,}<br>"
                "Units: %{customdata[2]:,}<br>"
                "Spend/Basket: $%{customdata[3]:,.2f}<br>"
                "ASP: $%{customdata[4]:,.2f}<br><extra></extra>"
            )
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=600)
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Metric definitions", expanded=False):
            st.markdown("""
- **Spend/Basket**: Total Sales ÷ Transactions.  
- **ASP**: Total Sales ÷ Units.
            """)

# -----------------------------------------------------------------------------
# Tab 7: Assortment & Space Optimization
# -----------------------------------------------------------------------------
if "📦 Assortment & Space" in VISIBLE_TABS:
    with tabs[VISIBLE_TABS.index("📦 Assortment & Space")]:
        st.markdown("### Assortment & Space Optimization")
        stores, cats, brands, prods, pays, rule, sd, ed = local_filter_block("aso")
        df_ = apply_local_filters(raw_df, stores, cats, brands, prods, pays, sd, ed)
        if df_.empty:
            st.info("No data for the selected filters.")
        else:
            # Weeks in range (anchored)
            try:
                n_weeks = max(int(df_.set_index("Date").resample("W-SUN")["Total_Sale"].sum().shape[0]),1)
            except:
                n_weeks = max(int(np.ceil((pd.to_datetime(ed)-pd.to_datetime(sd)).days/7.0)),1)

            # --- SKU Productivity
            st.subheader("A) SKU Productivity")
            prod_dim_opts = [c for c in ["Item","Brand_Name"] if c in df_.columns]
            prod_dim = st.selectbox("Granularity", prod_dim_opts, index=0, key="aso_dim")
            sku_store = (df_.groupby([prod_dim,"Store_ID"], dropna=False)["Total_Sale"].sum().reset_index())
            active_stores = sku_store.groupby(prod_dim)["Store_ID"].nunique().rename("Active_Stores")
            sku_sales = df_.groupby([prod_dim,"Category"], dropna=False)["Total_Sale"].sum().rename("SKU_Sales")
            cat_sales = df_.groupby("Category", dropna=False)["Total_Sale"].sum().rename("Category_Sales")
            prod_df = sku_sales.reset_index().merge(active_stores, on=prod_dim, how="left").merge(cat_sales.reset_index(), on="Category", how="left")
            prod_df["Active_Stores"] = prod_df["Active_Stores"].fillna(0).astype(int)
            prod_df["Weeks"] = n_weeks
            prod_df["Velocity_$/SKU/Store/Week"] = np.where(
                (prod_df["Active_Stores"]>0)&(prod_df["Weeks"]>0),
                prod_df["SKU_Sales"]/(prod_df["Active_Stores"]*prod_df["Weeks"]), 0.0
            )
            prod_df["% of Category Sales"] = np.where(
                prod_df["Category_Sales"]>0, 100.0*prod_df["SKU_Sales"]/prod_df["Category_Sales"], 0.0
            )
            prod_df["Velocity_Rank_in_Category"] = prod_df.groupby("Category")["Velocity_$/SKU/Store/Week"].rank(ascending=False, method="dense").astype(int)

            show_cols = [prod_dim,"Category","Active_Stores","Weeks","Velocity_$/SKU/Store/Week","SKU_Sales","Category_Sales","% of Category Sales","Velocity_Rank_in_Category"]
            view = prod_df[show_cols].sort_values(["Category","Velocity_$/SKU/Store/Week"], ascending=[True,False]).reset_index(drop=True)
            st.dataframe(
                view, hide_index=True, use_container_width=True,
                column_config={
                    prod_dim: st.column_config.TextColumn("SKU" if prod_dim=="Item" else "Brand"),
                    "Active_Stores": st.column_config.NumberColumn("Active Stores", format="%d",
                        help="Stores where the SKU sold at least once."),
                    "Weeks": st.column_config.NumberColumn("Weeks", format="%d", help="Anchored weekly buckets in range."),
                    "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f",
                        help="SKU Sales ÷ Active Stores ÷ Weeks."),
                    "SKU_Sales": st.column_config.NumberColumn("SKU Sales", format="$%.0f"),
                    "Category_Sales": st.column_config.NumberColumn("Category Sales", format="$%.0f"),
                    "% of Category Sales": st.column_config.ProgressColumn("% of Category Sales", min_value=0, max_value=100, format="%.2f%%"),
                    "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d")
                }
            )

            st.divider()

            # --- SKU Rationalization
            st.subheader("B) SKU Rationalization")
            st.caption("Adjust thresholds to flag low performers or near-duplicate SKUs. Any single flag marks an item.")
            c1,c2,c3 = st.columns(3)
            with c1:
                perc = st.slider("Low Velocity (percentile)", 5, 50, 25, step=5, key="aso_low")
            with c2:
                share_min = st.number_input("Min Category Share (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.5, key="aso_share")
            with c3:
                show_flagged = st.checkbox("Show only flagged", value=True, key="aso_only")
            rat = prod_df.copy()
            # flags
            for cat, grp in rat.groupby("Category"):
                cutoff = np.percentile(grp["Velocity_$/SKU/Store/Week"], perc) if len(grp) else 0.0
                idx = rat["Category"]==cat
                rat.loc[idx,"Low_Velocity_Flag"] = rat.loc[idx,"Velocity_$/SKU/Store/Week"] <= cutoff
            rat["Low_Share_Flag"] = rat["% of Category Sales"] < share_min
            rat["Rationalize?"] = rat["Low_Velocity_Flag"] | rat["Low_Share_Flag"]
            rview = rat[[prod_dim,"Category","Velocity_$/SKU/Store/Week","% of Category Sales",
                         "Velocity_Rank_in_Category","Low_Velocity_Flag","Low_Share_Flag","Rationalize?"]].copy()
            if show_flagged:
                rview = rview[rview["Rationalize?"]]
            rview = rview.sort_values(["Rationalize?","Category","Velocity_$/SKU/Store/Week"],
                                      ascending=[False,True,True]).reset_index(drop=True)
            st.dataframe(
                rview, hide_index=True, use_container_width=True,
                column_config={
                    prod_dim: st.column_config.TextColumn("SKU" if prod_dim=="Item" else "Brand"),
                    "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f"),
                    "% of Category Sales": st.column_config.NumberColumn("% of Category Sales", format="%.2f"),
                    "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d"),
                    "Low_Velocity_Flag": st.column_config.CheckboxColumn("Low Velocity"),
                    "Low_Share_Flag": st.column_config.CheckboxColumn("Low Share"),
                    "Rationalize?": st.column_config.CheckboxColumn("Flag")
                }
            )

            st.divider()

            # --- Opportunity Map
            st.subheader("C) Assortment Opportunity Map")
            geo_cols = [c for c in ["Store_State","Store_City","Store_ID"] if c in df_.columns]
            if not geo_cols:
                st.info("No geography columns available.")
            else:
                col1,col2,col3 = st.columns([1,1,1])
                with col1: geo = st.selectbox("Geo", geo_cols, key="aso_geo")
                with col2: md = st.selectbox("Analyze by", [c for c in ["Item","Brand_Name"] if c in df_.columns], key="aso_map_dim")
                with col3: topn = st.slider("Top-N per Geo", 3, 15, 5, key="aso_topn")
                overall_sales = df_.groupby(md)["Total_Sale"].sum().rename("Overall_Sales")
                overall_total = overall_sales.sum()
                overall_share = (overall_sales/overall_total).rename("Overall_Share").reset_index()
                geo_sales = df_.groupby([geo, md])["Total_Sale"].sum().rename("Geo_Sales").reset_index()
                geo_totals = df_.groupby(geo)["Total_Sale"].sum().rename("Geo_Total").reset_index()
                gs = geo_sales.merge(geo_totals, on=geo, how="left")
                gs["Geo_Share"] = np.where(gs["Geo_Total"]>0, gs["Geo_Sales"]/gs["Geo_Total"], 0.0)
                gs = gs.merge(overall_share, on=md, how="left")
                gs["Index_vs_Overall"] = np.where(gs["Overall_Share"]>0, 100.0*gs["Geo_Share"]/gs["Overall_Share"], 0.0)
                top_by_geo = (gs.sort_values("Index_vs_Overall", ascending=False).groupby(geo).head(topn).reset_index(drop=True))
                st.dataframe(
                    top_by_geo[[geo, md, "Index_vs_Overall","Geo_Share","Overall_Share","Geo_Sales"]],
                    hide_index=True, use_container_width=True,
                    column_config={
                        geo: st.column_config.TextColumn(geo.replace("_"," ")),
                        md: st.column_config.TextColumn(md.replace("_"," ")),
                        "Index_vs_Overall": st.column_config.NumberColumn("Index vs Overall", format="%.1f",
                            help="100 × Geo_Share ÷ Overall_Share; >100 = over-indexing."),
                        "Geo_Share": st.column_config.NumberColumn("Geo Share", format="%.2%"),
                        "Overall_Share": st.column_config.NumberColumn("Overall Share", format="%.2%"),
                        "Geo_Sales": st.column_config.NumberColumn("Geo Sales", format="$%.0f")
                    }
                )

            st.divider()

            # --- New Item Tracker
            st.subheader("D) New Item Tracker")
            if "Item" not in df_.columns:
                st.info("Item column required.")
            else:
                new_window = st.select_slider("Define 'New' (days since first sale)", options=[30,60,90,120,180], value=90, key="aso_newwin")
                all_first = raw_df.groupby("Item")["Date"].min().rename("First_Sale_Date").reset_index()
                cutoff = pd.to_datetime(ed) - pd.Timedelta(days=int(new_window))
                all_first["Is_New"] = all_first["First_Sale_Date"] >= cutoff
                f = df_.merge(all_first, on="Item", how="left")
                new_items = f[f["Is_New"]==True].copy()
                if new_items.empty:
                    st.info("No newly introduced items in the current selection.")
                else:
                    perf = (f.groupby(["Item","Category"], dropna=False)["Total_Sale"].sum().rename("Item_Sales").reset_index()
                              .merge(f.groupby(["Item"])["Store_ID"].nunique().rename("Active_Stores").reset_index(), on="Item", how="left"))
                    perf["Weeks"] = n_weeks
                    perf["Velocity_$/SKU/Store/Week"] = np.where(
                        (perf["Active_Stores"]>0)&(perf["Weeks"]>0),
                        perf["Item_Sales"]/(perf["Active_Stores"]*perf["Weeks"]), 0.0
                    )
                    new_perf = new_items[["Item","Category"]].drop_duplicates().merge(perf, on=["Item","Category"], how="left")
                    # benchmark = non-new in same category
                    bench = f.merge(all_first[all_first["Is_New"]==False][["Item"]].assign(Not_New=True), on="Item", how="inner")
                    bench_perf = (bench.groupby(["Item","Category"], dropna=False)["Total_Sale"].sum().rename("Item_Sales").reset_index()
                                     .merge(bench.groupby(["Item"])["Store_ID"].nunique().rename("Active_Stores").reset_index(), on="Item", how="left"))
                    bench_perf["Weeks"]=n_weeks
                    bench_perf["Velocity_$/SKU/Store/Week"] = np.where(
                        (bench_perf["Active_Stores"]>0)&(bench_perf["Weeks"]>0),
                        bench_perf["Item_Sales"]/(bench_perf["Active_Stores"]*bench_perf["Weeks"]), 0.0
                    )
                    bench_avg = bench_perf.groupby("Category")["Velocity_$/SKU/Store/Week"].mean().rename("Benchmark_Velocity").reset_index()
                    new_perf = new_perf.merge(bench_avg, on="Category", how="left")
                    new_perf["Velocity_vs_Benchmark_%"] = np.where(
                        new_perf["Benchmark_Velocity"]>0,
                        100.0*new_perf["Velocity_$/SKU/Store/Week"]/new_perf["Benchmark_Velocity"], np.nan
                    )
                    new_perf = new_perf.sort_values(["Category","Velocity_vs_Benchmark_%"], ascending=[True,False]).reset_index(drop=True)
                    st.dataframe(
                        new_perf[["Item","Category","Active_Stores","Weeks","Velocity_$/SKU/Store/Week","Benchmark_Velocity","Velocity_vs_Benchmark_%"]],
                        hide_index=True, use_container_width=True,
                        column_config={
                            "Active_Stores": st.column_config.NumberColumn("Active Stores", format="%d"),
                            "Weeks": st.column_config.NumberColumn("Weeks", format="%d"),
                            "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f"),
                            "Benchmark_Velocity": st.column_config.NumberColumn("Benchmark Velocity", format="$%.2f"),
                            "Velocity_vs_Benchmark_%": st.column_config.NumberColumn("Velocity vs Benchmark (%)", format="%.1f")
                        }
                    )

# -----------------------------------------------------------------------------
# Small data quality panel
# -----------------------------------------------------------------------------
with st.expander("Data validation summary", expanded=False):
    rows = len(raw_df); baskets = raw_df["Transaction_ID"].nunique()
    st.caption(f"Rows: **{rows:,}** | Baskets: **{baskets:,}** | Date Range: **{DOMAIN['min_date']} → {DOMAIN['max_date']}**")
    issues=[]
    if "Quantity" in raw_df.columns and (raw_df["Quantity"]<=0).any(): issues.append("Non-positive Quantity")
    if "Total_Sale" in raw_df.columns and (raw_df["Total_Sale"]<0).any(): issues.append("Negative Total_Sale")
    if issues: st.warning(" | ".join(issues))
    else: st.success("No obvious data issues detected.")
