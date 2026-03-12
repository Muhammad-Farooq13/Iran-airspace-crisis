"""
Iran Airspace Crisis — Streamlit Dashboard
==========================================
4 tabs:
  1. Crisis Overview   — timeline, map, key metrics
  2. Airline Loss Predictor — interactive ML prediction
  3. Data Explorer     — raw CSV browser + charts
  4. Pipeline & Models — architecture, model comparison
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iran Airspace Crisis — Analysis Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
RAW_DIR    = ROOT / "data" / "raw"
DEMO_PKL   = ROOT / "models" / "iran_demo.pkl"
RESULTS_JSON = ROOT / "models" / "training_results.json"

# ─── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / name)


@st.cache_resource
def load_demo_model():
    with open(DEMO_PKL, "rb") as f:
        return pickle.load(f)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Flag_of_Iran.svg/1200px-Flag_of_Iran.svg.png",
        width=110,
    )
    st.markdown("## Iran Airspace Crisis")
    st.markdown(
        """
**Crisis start:** 28 Feb 2026  
**Trigger:** US airstrikes on Iranian nuclear facilities (Natanz, Fordow, Arak)  
**Impact:** Iranian FIR closure, cascading regional disruption  

---
*Data spans 12 days of the acute crisis phase (Feb 28 – Mar 10, 2026)*
"""
    )
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · Plotly")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Crisis Overview", "✈️ Airline Loss Predictor", "🔍 Data Explorer", "⚙️ Pipeline & Models"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CRISIS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🗺️ Iran Airspace Crisis — Overview")
    st.markdown(
        """
On **28 February 2026**, US airstrikes on Iranian nuclear facilities at Natanz, Fordow, and Arak  
triggered the **immediate closure of Iranian airspace (OIIX FIR)** and a cascade of regional airspace  
restrictions, emergency NOTAMs, and flight diversions affecting carriers across Europe, Asia, and the Gulf.
"""
    )

    # ── Key metrics ──────────────────────────────────────────────────────────
    losses_df  = load_raw("airline_losses_estimate.csv")
    cancel_df  = load_raw("flight_cancellations.csv")
    reroute_df = load_raw("flight_reroutes.csv")
    conflict_df = load_raw("conflict_events.csv")

    total_loss      = losses_df["estimated_daily_loss_usd"].sum()
    total_cancelled = losses_df["cancelled_flights"].sum()
    total_rerouted  = losses_df["rerouted_flights"].sum()
    total_pax       = losses_df["passengers_impacted"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Estimated Losses", f"${total_loss/1e6:.1f}M")
    c2.metric("Flights Cancelled",      f"{total_cancelled:,}")
    c3.metric("Flights Rerouted",       f"{total_rerouted:,}")
    c4.metric("Passengers Impacted",    f"{total_pax:,}")

    st.markdown("---")

    col_left, col_right = st.columns([1.3, 1])

    # ── Conflict event map ────────────────────────────────────────────────────
    with col_left:
        st.subheader("Conflict Event Locations")
        sev_map = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
        conflict_df["sev_num"] = conflict_df["severity"].map(sev_map).fillna(1)

        fig_map = px.scatter_mapbox(
            conflict_df,
            lat="latitude", lon="longitude",
            color="severity",
            size="sev_num",
            size_max=20,
            hover_name="location",
            hover_data={"event_type": True, "aviation_impact": True, "date": True,
                        "sev_num": False, "latitude": False, "longitude": False},
            color_discrete_map={
                "Critical": "#e74c3c",
                "High":     "#e67e22",
                "Medium":   "#f1c40f",
                "Low":      "#2ecc71",
            },
            zoom=3.5,
            center={"lat": 32, "lon": 53},
            mapbox_style="carto-darkmatter",
            title="Conflict events — Feb/Mar 2026",
            height=420,
        )
        fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0},
                              legend_title_text="Severity")
        st.plotly_chart(fig_map, use_container_width=True)

    # ── Top-hit airlines ─────────────────────────────────────────────────────
    with col_right:
        st.subheader("Top Airlines by Estimated Loss")
        top10 = losses_df.nlargest(10, "estimated_daily_loss_usd")
        fig_bar = px.bar(
            top10,
            x="estimated_daily_loss_usd",
            y="airline",
            orientation="h",
            color="estimated_daily_loss_usd",
            color_continuous_scale="Reds",
            labels={"estimated_daily_loss_usd": "Estimated Loss (USD)", "airline": ""},
            height=420,
        )
        fig_bar.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Conflict events timeline ──────────────────────────────────────────────
    st.subheader("Conflict Events Timeline")
    conflict_df["date"] = pd.to_datetime(conflict_df["date"])
    timeline_df = conflict_df.sort_values("date")
    fig_tl = px.scatter(
        timeline_df,
        x="date",
        y="location",
        color="event_type",
        symbol="severity",
        size="sev_num",
        size_max=18,
        hover_data={"aviation_impact": True, "source": True, "sev_num": False},
        labels={"date": "Date", "location": "Location", "event_type": "Event Type"},
        height=340,
    )
    fig_tl.update_layout(
        margin={"t": 20, "b": 0},
        xaxis_title="Date",
        yaxis_title="",
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # ── Airspace closures ─────────────────────────────────────────────────────
    st.subheader("Airspace Closures Summary")
    closure_df = load_raw("airspace_closures.csv")
    st.dataframe(closure_df, use_container_width=True, height=200)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AIRLINE LOSS PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("✈️ Airline Daily Loss Predictor")
    st.markdown(
        """
Estimate the **daily revenue loss (USD)** for an airline operating during the crisis,  
based on its operational exposure. The model is a **Gradient Boosting Regressor**  
trained on crisis-period airline data (Test R² = 0.96).
"""
    )

    demo = load_demo_model()
    model   = demo["model"]
    scaler  = demo["scaler"]
    features = demo["features"]
    country_cols = demo["country_cols"]
    metrics = demo["metrics"]
    all_countries = sorted(c.replace("country_", "") for c in country_cols)

    with st.expander("📊 Demo Model Metrics", expanded=False):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Test R²",    f"{metrics['test_r2']:.4f}")
        mc2.metric("Test MAE",   f"${metrics['test_mae']:,.0f}")
        mc3.metric("CV R² Mean", f"{metrics['cv_r2_mean']:.4f}")
        mc4.metric("CV R² Std",  f"±{metrics['cv_r2_std']:.4f}")

    st.markdown("---")
    st.subheader("Enter Airline Operational Parameters")

    col1, col2 = st.columns(2)
    with col1:
        cancelled   = st.slider("Cancelled flights",          0,  50, 10)
        rerouted    = st.slider("Rerouted flights",           0, 100, 30)
        fuel_cost   = st.number_input("Additional fuel cost (USD)", min_value=0,
                                      max_value=10_000_000, value=1_500_000, step=50_000)
        pax         = st.number_input("Passengers impacted",  min_value=0,
                                      max_value=50_000, value=5_000, step=100)
    with col2:
        country     = st.selectbox("Airline home country", ["(select)"] + all_countries, index=0)
        st.info(
            "**Derived features (auto-calculated)**\n\n"
            "- Fuel cost ratio = fuel_cost / (cancelled + rerouted + 1)\n"
            "- Reroute ratio = rerouted / (cancelled + rerouted + 1)\n"
            "- Log fuel cost = log(1 + fuel_cost)"
        )

    predict_btn = st.button("🔮 Predict Daily Loss", type="primary", use_container_width=True)

    if predict_btn:
        if country == "(select)":
            st.warning("Please select an airline home country.")
        else:
            total_ops      = cancelled + rerouted + 1
            fuel_cost_ratio = fuel_cost / total_ops
            reroute_ratio   = rerouted / total_ops
            log_fuel        = np.log1p(fuel_cost)

            # Build input row aligned to feature list
            row = {f: 0.0 for f in features}
            row["cancelled_flights"]           = float(cancelled)
            row["rerouted_flights"]            = float(rerouted)
            row["additional_fuel_cost_usd"]    = float(fuel_cost)
            row["passengers_impacted"]         = float(pax)
            row["fuel_cost_ratio"]             = fuel_cost_ratio
            row["reroute_ratio"]               = reroute_ratio
            row["log_fuel_cost"]               = log_fuel
            country_key = f"country_{country}"
            if country_key in row:
                row[country_key] = 1.0

            X_in = pd.DataFrame([row])[features]
            X_sc = scaler.transform(X_in)
            pred = float(model.predict(X_sc)[0])
            pred = max(pred, 0.0)

            st.markdown("---")
            res1, res2, res3 = st.columns(3)
            res1.metric("Predicted Daily Loss", f"${pred:,.0f}")
            res2.metric("In Millions",          f"${pred/1e6:.3f}M")
            res3.metric("Per Passenger",        f"${pred/max(pax,1):,.0f}")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred / 1e6,
                title={"text": "Predicted Loss (USD millions)", "font": {"size": 18}},
                delta={"reference": 1.5, "relative": False,
                       "decreasing": {"color": "green"}, "increasing": {"color": "red"}},
                gauge={
                    "axis": {"range": [0, 10], "tickformat": ".1f"},
                    "bar": {"color": "#e74c3c"},
                    "steps": [
                        {"range": [0, 1],   "color": "rgba(46,204,113,0.25)"},
                        {"range": [1, 3],   "color": "rgba(241,196,15,0.25)"},
                        {"range": [3, 6],   "color": "rgba(230,126,34,0.25)"},
                        {"range": [6, 10],  "color": "rgba(231,76,60,0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": pred / 1e6,
                    },
                },
                number={"suffix": "M", "valueformat": ".3f"},
            ))
            fig_gauge.update_layout(height=300, margin={"t": 30, "b": 10, "l": 30, "r": 30})
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Severity annotation
            if pred < 500_000:
                st.success("🟢 **Low impact** — minimal route disruption.")
            elif pred < 2_000_000:
                st.warning("🟡 **Moderate impact** — significant cancellations and rerouting costs.")
            elif pred < 5_000_000:
                st.error("🟠 **High impact** — major network disruption, substantial passenger impact.")
            else:
                st.error("🔴 **Severe impact** — crisis-level losses. Immediate operational changes advised.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🔍 Data Explorer")

    DATASETS = {
        "Airline Losses":       "airline_losses_estimate.csv",
        "Conflict Events":      "conflict_events.csv",
        "Airspace Closures":    "airspace_closures.csv",
        "Airport Disruptions":  "airport_disruptions.csv",
        "Flight Cancellations": "flight_cancellations.csv",
        "Flight Reroutes":      "flight_reroutes.csv",
    }

    selected = st.selectbox("Choose dataset", list(DATASETS.keys()))
    df_exp = load_raw(DATASETS[selected])

    st.markdown(f"**{len(df_exp):,} rows × {len(df_exp.columns)} columns**")
    st.dataframe(df_exp, use_container_width=True)

    st.markdown("---")
    numeric_cols = df_exp.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) >= 1:
        col_a, col_b = st.columns(2)
        with col_a:
            x_col = st.selectbox("X-axis column", numeric_cols, key="x_col")
            fig_hist = px.histogram(df_exp, x=x_col, nbins=20,
                                    title=f"Distribution of {x_col}",
                                    color_discrete_sequence=["#e74c3c"])
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            if len(numeric_cols) >= 2:
                y_col = st.selectbox("Y-axis column", numeric_cols,
                                     index=min(1, len(numeric_cols)-1), key="y_col")
                fig_sc = px.scatter(df_exp, x=x_col, y=y_col,
                                    title=f"{x_col} vs {y_col}",
                                    trendline="ols" if len(df_exp) > 4 else None,
                                    color_discrete_sequence=["#3498db"])
                st.plotly_chart(fig_sc, use_container_width=True)

    if len(numeric_cols) >= 3:
        st.subheader("Correlation Heatmap")
        corr = df_exp[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            title="Pearson Correlation Matrix",
            height=max(300, 60 * len(numeric_cols)),
        )
        fig_corr.update_layout(margin={"t": 50, "b": 10})
        st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PIPELINE & MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("⚙️ ML Pipeline & Model Comparison")

    st.subheader("Pipeline Architecture")
    st.markdown(
        """
```
data/raw/*.csv
    │
    ▼  src/data/pipeline.py      ── clean, parse dates, derive ratios, join
data/processed/master_dataset.parquet
    │
    ▼  src/features/build_features.py  ── 22 numeric + country one-hot → 55 features
data/features/feature_matrix.parquet  +  models/scaler.pkl  +  models/feature_list.json
    │
    ▼  src/models/train.py       ── KFold CV over 4 candidate models, save best
models/airline_loss_regressor.pkl      (best: Ridge, CV R² = 0.892)
    │
    ▼  src/api/app.py            ── FastAPI REST service (port 8000)
POST /v1/predict   →   {"predicted_daily_loss_usd": ..., "predicted_daily_loss_millions": ...}
```
"""
    )

    # Model comparison
    st.subheader("Model Comparison (full-pipeline training)")
    results_path = RESULTS_JSON
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        rows = []
        for name, res in results["results"].items():
            rows.append({
                "Model":         name.replace("_", " ").title(),
                "Train R²":      res["train"]["r2"],
                "Test R²":       res["test"]["r2"],
                "Test RMSE":     res["test"]["rmse"],
                "Test MAE":      res["test"]["mae"],
                "CV R² Mean":    res["cv"]["cv_r2_mean"],
                "CV R² Std":     res["cv"]["cv_r2_std"],
                "Best":          "✅" if name == results["best_model"] else "",
            })

        cmp_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(
            cmp_df.style.format({
                "Train R²":   "{:.4f}",
                "Test R²":    "{:.4f}",
                "Test RMSE":  "${:,.0f}",
                "Test MAE":   "${:,.0f}",
                "CV R² Mean": "{:.4f}",
                "CV R² Std":  "{:.4f}",
            }).highlight_max(subset=["Test R²", "CV R² Mean"], color="rgba(46,204,113,0.35)")
             .highlight_min(subset=["Test RMSE", "Test MAE"], color="rgba(46,204,113,0.35)"),
            use_container_width=True,
        )

        # Bar chart
        cmp_chart = pd.DataFrame([
            {"Model": r["Model"], "Metric": "Test R²",    "Value": r["Test R²"]}
            for r in rows
        ] + [
            {"Model": r["Model"], "Metric": "CV R² Mean", "Value": r["CV R² Mean"]}
            for r in rows
        ])
        fig_cmp = px.bar(
            cmp_chart,
            x="Model", y="Value", color="Metric",
            barmode="group",
            title="Model R² Comparison",
            color_discrete_sequence=["#3498db", "#e74c3c"],
            height=350,
        )
        fig_cmp.update_layout(yaxis={"range": [0, 1.05]})
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Run `python -m src.models.train` to generate training_results.json.")

    # Feature list
    st.subheader("Full Feature Set (55 features)")
    feature_cols_json = ROOT / "models" / "feature_list.json"
    if feature_cols_json.exists():
        with open(feature_cols_json) as f:
            full_features = json.load(f)
        numeric_feats   = [f for f in full_features if not f.startswith("country_")]
        country_feats   = [f for f in full_features if f.startswith("country_")]
        fdf = pd.DataFrame({
            "Numeric features": pd.Series(numeric_feats),
            "Country dummies":  pd.Series(country_feats),
        })
        st.dataframe(fdf.fillna(""), use_container_width=True)

    # API docs link
    st.subheader("REST API")
    st.markdown(
        """
Start the FastAPI service:
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```
Then browse interactive docs at **http://localhost:8000/docs**

| Endpoint                   | Method | Description                  |
|----------------------------|--------|------------------------------|
| `/`                        | GET    | Health check                 |
| `/v1/info`                 | GET    | Model metadata & feature list|
| `/v1/predict`              | POST   | Single prediction            |
| `/v1/predict/batch`        | POST   | Batch predictions (≤ 500)    |
"""
    )
