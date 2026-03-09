"""
Reusable visualization functions for EDA and reporting.

All functions accept a matplotlib Axes object (or create their own) and
return it, making them composable in both notebooks and scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = sns.color_palette("muted")
RISK_PALETTE = {
    "CRITICAL": "#d62728",
    "HIGH":     "#ff7f0e",
    "MEDIUM":   "#2ca02c",
    "LOW":      "#1f77b4",
}

plt.rcParams.update({
    "figure.dpi":     150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":    "DejaVu Sans",
})


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Airline losses
# ──────────────────────────────────────────────────────────────────────────────

def plot_airline_losses(df: pd.DataFrame, top_n: int = 15, save: bool = True) -> plt.Figure:
    """Horizontal bar chart – top airlines by estimated daily loss."""
    data = df.nlargest(top_n, "estimated_daily_loss_usd").sort_values("estimated_daily_loss_usd")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(data["airline"], data["estimated_daily_loss_usd"] / 1e6, color=PALETTE[0])
    ax.bar_label(bars, fmt="$%.1fM", padding=4, fontsize=9)
    ax.set_xlabel("Estimated Daily Loss (USD million)")
    ax.set_title(f"Top {top_n} Airlines by Daily Loss — Iran Airspace Crisis", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
    plt.tight_layout()
    if save:
        _save(fig, "airline_daily_losses_bar")
    return fig


def plot_loss_breakdown(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Stacked bar: fuel cost vs other losses per airline."""
    df2 = df.nlargest(12, "estimated_daily_loss_usd").copy()
    df2["other_loss"] = df2["estimated_daily_loss_usd"] - df2["additional_fuel_cost_usd"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(df2))
    ax.bar(x, df2["additional_fuel_cost_usd"] / 1e6, label="Fuel Surcharge", color=PALETTE[1])
    ax.bar(x, df2["other_loss"] / 1e6, bottom=df2["additional_fuel_cost_usd"] / 1e6,
           label="Other Revenue Loss", color=PALETTE[0])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df2["airline"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("USD Million/day")
    ax.set_title("Loss Breakdown: Fuel Surcharge vs. Other Revenue Loss", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save:
        _save(fig, "loss_breakdown_stacked")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Conflict events timeline
# ──────────────────────────────────────────────────────────────────────────────

def plot_conflict_timeline(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Scatter plot of conflict events over time, colored by severity."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for sev, grp in df.groupby("severity"):
        color = RISK_PALETTE.get(sev, "grey")
        size  = grp["severity_score"] * 40
        ax.scatter(grp["datetime_utc"], grp.index, c=color, s=size,
                   label=sev, alpha=0.85, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Date (UTC)")
    ax.set_yticks([])
    ax.set_title("Conflict Event Timeline — Severity Over Time", fontweight="bold")
    ax.legend(title="Severity", loc="upper right")
    fig.autofmt_xdate()
    plt.tight_layout()
    if save:
        _save(fig, "conflict_event_timeline")
    return fig


def plot_severity_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Pie chart of conflict event severity distribution."""
    counts = df["severity"].value_counts()
    colors = [RISK_PALETTE.get(s, "grey") for s in counts.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.8,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Conflict Events by Severity", fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "conflict_severity_pie")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Airspace closures
# ──────────────────────────────────────────────────────────────────────────────

def plot_closure_duration(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Horizontal bars showing airspace closure duration per country/FIR."""
    df_sorted = df.sort_values("closure_duration_hours", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = df_sorted["closure_type"].map({
        "active_conflict": RISK_PALETTE["CRITICAL"],
        "precautionary":   RISK_PALETTE["MEDIUM"],
        "spillover":       RISK_PALETTE["HIGH"],
        "other":           "steelblue",
    }).fillna("steelblue")
    ax.barh(
        df_sorted["country"] + " — " + df_sorted["region"],
        df_sorted["closure_duration_hours"],
        color=colors,
    )
    ax.set_xlabel("Closure Duration (hours)")
    ax.set_title("Airspace Closure Duration by Country/FIR", fontweight="bold")
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=RISK_PALETTE["CRITICAL"], label="Active Conflict"),
        Patch(facecolor=RISK_PALETTE["HIGH"],     label="Spillover"),
        Patch(facecolor=RISK_PALETTE["MEDIUM"],   label="Precautionary"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    if save:
        _save(fig, "airspace_closure_duration")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Airport disruptions
# ──────────────────────────────────────────────────────────────────────────────

def plot_airport_disruptions(df: pd.DataFrame, top_n: int = 20, save: bool = True) -> plt.Figure:
    """Grouped bar – cancelled, delayed, diverted per airport."""
    data = df.nlargest(top_n, "total_disrupted")
    x = np.arange(len(data))
    width = 0.28

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, data["flights_cancelled"], width, label="Cancelled", color=PALETTE[3])
    ax.bar(x,         data["flights_delayed"],   width, label="Delayed",   color=PALETTE[1])
    ax.bar(x + width, data["flights_diverted"],  width, label="Diverted",  color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(data["iata"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Number of Flights")
    ax.set_title(f"Top {top_n} Airports by Flight Disruptions", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save:
        _save(fig, "airport_disruptions_grouped")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Reroutes
# ──────────────────────────────────────────────────────────────────────────────

def plot_reroute_cost_vs_distance(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Scatter: additional distance vs fuel cost, annotated by airline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        df["additional_distance_km"],
        df["additional_fuel_cost_usd"] / 1000,
        c=df["delay_minutes"],
        cmap="YlOrRd",
        s=60,
        alpha=0.85,
        edgecolors="grey",
        linewidths=0.4,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Delay (minutes)")
    ax.set_xlabel("Additional Distance (km)")
    ax.set_ylabel("Additional Fuel Cost (USD thousands)")
    ax.set_title("Reroute Impact: Distance vs. Fuel Cost vs. Delay", fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "reroute_cost_distance_scatter")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Correlation heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, cols: Optional[list[str]] = None,
                             save: bool = True) -> plt.Figure:
    """Seaborn heatmap of numeric feature correlations."""
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "correlation_heatmap")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ──────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs. Predicted Daily Loss",
    save: bool = True,
) -> plt.Figure:
    """45-degree line plot to assess regression fit quality."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true / 1e6, y_pred / 1e6, alpha=0.75, color=PALETTE[0],
               edgecolors="white", linewidths=0.5)
    lo = min(y_true.min(), y_pred.min()) / 1e6
    hi = max(y_true.max(), y_pred.max()) / 1e6
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Loss (USD Million)")
    ax.set_ylabel("Predicted Loss (USD Million)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save:
        _save(fig, "actual_vs_predicted")
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 20, save: bool = True) -> plt.Figure:
    """Horizontal bar chart of top feature importances."""
    data = fi_df.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["feature"], data["importance"], color=PALETTE[2])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "feature_importances")
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Residual plot to check heteroscedasticity."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs fitted
    axes[0].scatter(y_pred / 1e6, residuals / 1e6, alpha=0.75, color=PALETTE[0],
                    edgecolors="white", linewidths=0.4)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted (USD Million)")
    axes[0].set_ylabel("Residual (USD Million)")
    axes[0].set_title("Residuals vs. Fitted", fontweight="bold")

    # Residual distribution
    axes[1].hist(residuals / 1e6, bins=15, color=PALETTE[2], edgecolor="white")
    axes[1].set_xlabel("Residual (USD Million)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution", fontweight="bold")

    plt.tight_layout()
    if save:
        _save(fig, "residual_analysis")
    return fig
