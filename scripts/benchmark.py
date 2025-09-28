from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try seaborn for nicer plots; fall back gracefully
try:
    import seaborn as sns  # type: ignore
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# Project paths (scripts folder -> project root is parents[1])
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "Results"
EMISSIONS_DIR = RESULTS_DIR / "emissions"
REPORTS_DIR = RESULTS_DIR / "reports"
BENCHMARK_DIR = RESULTS_DIR / "Benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# Focus countries for comparison overlay
FOCUS_COUNTRIES = {"Denmark", "Germany", "Hungary", "Spain"}


def _is_emissions_csv(p: Path) -> bool:
    return p.suffix.lower() == ".csv" and p.name.endswith("_emissions.csv")


def _parse_ctx_from_path(p: Path) -> Tuple[str, str, str, str]:
    # Expect: Results/emissions/<Model>/<Country>/<season>/<run>_emissions.csv
    # Return: (model, country, season, run_name)
    parts = p.parts
    # Find the index of 'emissions'
    try:
        idx = parts.index("emissions")
    except ValueError:
        # try full path scan for folder name
        for i, s in enumerate(parts):
            if s.lower() == "emissions":
                idx = i
                break
        else:
            raise ValueError(f"Unrecognized emissions path structure: {p}")
    model = parts[idx + 1]
    country = parts[idx + 2]
    season = parts[idx + 3]
    run_file = parts[idx + 4]
    run_name = run_file.replace("_emissions.csv", "")
    return model, country, season, run_name


def _load_emissions_total(csv_path: Path) -> Dict[str, float]:
    # CodeCarbon writes time-series rows with 'emissions' and possibly energy/duration
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {"emissions_kg": np.nan, "energy_kwh": np.nan, "duration_sec": np.nan}

    # Columns may include: emissions, duration, energy_consumed, cpu_energy, gpu_energy
    emissions_col = None
    for c in df.columns:
        if c.lower() == "emissions":
            emissions_col = c
            break
    if emissions_col is not None:
        # If cumulative, take max; else sum
        vals = pd.to_numeric(df[emissions_col], errors="coerce").fillna(0.0).values
        if len(vals) == 0:
            total_emissions = 0.0
        else:
            is_non_decreasing = np.all(np.diff(vals) >= -1e-9)
            total_emissions = float(np.nanmax(vals) if is_non_decreasing else np.nansum(vals))
    else:
        total_emissions = float("nan")

    energy_col = None
    for c in ("energy_consumed", "energy_kwh", "energy_consumed_kwh"):
        if c in df.columns:
            energy_col = c
            break
    if energy_col is not None:
        energy_vals = pd.to_numeric(df[energy_col], errors="coerce").fillna(0.0).values
        is_non_decreasing_e = np.all(np.diff(energy_vals) >= -1e-9)
        total_energy = float(np.nanmax(energy_vals) if is_non_decreasing_e else np.nansum(energy_vals))
    else:
        total_energy = float("nan")

    # Aggregate duration if available
    duration_col = None
    for c in ("duration", "duration_sec"):
        if c in df.columns:
            duration_col = c
            break
    if duration_col is not None:
        duration_vals = pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0).values
        total_duration = float(np.nanmax(duration_vals))
    else:
        total_duration = float("nan")

    return {"emissions_kg": total_emissions, "energy_kwh": total_energy, "duration_sec": total_duration}


def collect_emissions() -> pd.DataFrame:
    rows: List[Dict] = []
    if not EMISSIONS_DIR.exists():
        return pd.DataFrame(columns=["model", "country", "season", "run", "emissions_kg", "energy_kwh", "duration_sec"]) 
    for model_dir in EMISSIONS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        for country_dir in model_dir.iterdir():
            if not country_dir.is_dir():
                continue
            for season_dir in country_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                for csv_path in season_dir.glob("*_emissions.csv"):
                    try:
                        model, country, season, run_name = _parse_ctx_from_path(csv_path)
                    except Exception:
                        # Skip unknown structure
                        continue
                    agg = _load_emissions_total(csv_path)
                    rows.append({
                        "model": model,
                        "country": country,
                        "season": season,
                        "run": run_name,
                        **agg,
                        "csv_path": str(csv_path),
                    })
    return pd.DataFrame(rows)


def save_aggregated(df: pd.DataFrame) -> Path:
    out = BENCHMARK_DIR / "emissions_aggregated.csv"
    df.to_csv(out, index=False)
    return out


def plot_box_all_models(df: pd.DataFrame) -> Path:
    # Single figure: x = model (41), y = emissions_kg, with jitter points colored by country
    fig, ax = plt.subplots(figsize=(max(12, 0.35 * max(1, df["model"].nunique())), 6))
    models_sorted = sorted(df["model"].unique().tolist())
    # Prepare data in order
    data = [df.loc[df["model"] == m, "emissions_kg"].dropna().values for m in models_sorted]
    ax.boxplot(data, labels=models_sorted, showfliers=False)
    # Overlay jitter points (countries colored)
    countries = sorted(df["country"].unique().tolist())
    # Use a version-compatible colormap getter (no N argument for older mpl)
    try:
        cmap = plt.colormaps.get_cmap('tab20')  # matplotlib >= 3.6
    except Exception:
        cmap = plt.cm.get_cmap('tab20')  # older matplotlib fallback
    x_positions = {m: i + 1 for i, m in enumerate(models_sorted)}
    for ci, country in enumerate(countries):
        sub = df[df["country"] == country]
        xs = [x_positions[m] for m in sub["model"]]
        ys = sub["emissions_kg"].values
        jitter = (np.random.rand(len(xs)) - 0.5) * 0.2
        color = cmap(ci % (getattr(cmap, 'N', 20) or 20))
        ax.scatter(np.array(xs) + jitter, ys, s=12, alpha=0.6, color=color, label=country if country in FOCUS_COUNTRIES else None)
    # Legend only for focus countries to avoid clutter
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, title="Country (focus)")
    ax.set_title("Carbon emissions per training — All models (box = distribution, dots = runs)")
    ax.set_xlabel("Model (41 variants)")
    ax.set_ylabel("Emissions (kg CO₂e)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    out = BENCHMARK_DIR / "boxplot_emissions_all_models.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_heatmaps(df: pd.DataFrame) -> List[Path]:
    outs: List[Path] = []
    for season in sorted(df["season"].unique().tolist()):
        piv = df[df["season"] == season].pivot_table(index="model", columns="country", values="emissions_kg", aggfunc="mean")
        fig_w = max(10, 0.4 * max(1, piv.shape[1]))
        fig_h = max(8, 0.3 * max(1, piv.shape[0]))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        if HAS_SNS:
            sns.heatmap(piv, annot=False, cmap="YlOrRd", ax=ax)
        else:
            im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels(piv.columns, rotation=90)
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels(piv.index)
            fig.colorbar(im, ax=ax)
        ax.set_title(f"Mean emissions (kg) by Model × Country — {season}")
        ax.set_xlabel("Country")
        ax.set_ylabel("Model")
        plt.tight_layout()
        out = BENCHMARK_DIR / f"heatmap_mean_emissions_{season}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_per_country_box(df: pd.DataFrame) -> List[Path]:
    outs: List[Path] = []
    for country in sorted(set(df["country"]) & FOCUS_COUNTRIES):
        sub = df[df["country"] == country]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(12, 0.35 * max(1, sub["model"].nunique())), 6))
        models_sorted = sorted(sub["model"].unique().tolist())
        data = [sub.loc[sub["model"] == m, "emissions_kg"].dropna().values for m in models_sorted]
        ax.boxplot(data, labels=models_sorted, showfliers=False)
        ax.set_title(f"Emissions per training — {country}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Emissions (kg CO₂e)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        out = BENCHMARK_DIR / f"boxplot_emissions_{country.replace(' ', '_')}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outs.append(out)
    return outs


# ---------------- Additional metrics aggregation and plots ---------------- #

def collect_metrics_from_reports() -> pd.DataFrame:
    rows: List[Dict] = []
    if not REPORTS_DIR.exists():
        return pd.DataFrame(columns=["model", "country", "season", "run", "mae", "rmse", "mse", "r2", "mape", "emissions_csv"]) 
    for model_dir in REPORTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        for country_dir in model_dir.iterdir():
            if not country_dir.is_dir():
                continue
            for season_dir in country_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                for report_path in season_dir.glob("*.json"):
                    run_name = report_path.stem
                    try:
                        data = json_load(report_path)
                    except Exception:
                        continue
                    metrics = data.get("metrics", {}) or {}
                    rows.append({
                        "model": model_dir.name,
                        "country": country_dir.name,
                        "season": season_dir.name,
                        "run": run_name,
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "mse": metrics.get("mse"),
                        "r2": metrics.get("r2"),
                        "mape": metrics.get("mape"),
                        "emissions_csv": data.get("emissions_csv"),
                        "report_path": str(report_path),
                    })
    return pd.DataFrame(rows)


def json_load(path: Path) -> Dict:
    import json
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics_aggregated(df: pd.DataFrame) -> Path:
    out = BENCHMARK_DIR / "metrics_aggregated.csv"
    df.to_csv(out, index=False)
    return out


def plot_metric_box_all_models(dfm: pd.DataFrame, metric: str) -> Path:
    d = dfm.dropna(subset=[metric])
    if d.empty:
        return BENCHMARK_DIR / f"boxplot_{metric}_all_models_EMPTY.png"
    fig, ax = plt.subplots(figsize=(max(12, 0.35 * max(1, d["model"].nunique())), 6))
    models_sorted = sorted(d["model"].unique().tolist())
    data = [d.loc[d["model"] == m, metric].astype(float).values for m in models_sorted]
    ax.boxplot(data, labels=models_sorted, showfliers=False)
    ax.set_title(f"{metric.upper()} per training — All models")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    plt.xticks(rotation=90)
    plt.tight_layout()
    out = BENCHMARK_DIR / f"boxplot_{metric}_all_models.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_metric_heatmaps(dfm: pd.DataFrame, metric: str) -> List[Path]:
    outs: List[Path] = []
    d = dfm.dropna(subset=[metric])
    for season in sorted(d["season"].unique().tolist()):
        piv = d[d["season"] == season].pivot_table(index="model", columns="country", values=metric, aggfunc="mean")
        if piv.empty:
            continue
        fig_w = max(10, 0.4 * max(1, piv.shape[1]))
        fig_h = max(8, 0.3 * max(1, piv.shape[0]))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        if HAS_SNS:
            sns.heatmap(piv, annot=False, cmap="Blues", ax=ax)
        else:
            im = ax.imshow(piv.values, aspect="auto", cmap="Blues")
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels(piv.columns, rotation=90)
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels(piv.index)
            fig.colorbar(im, ax=ax)
        ax.set_title(f"Mean {metric.upper()} by Model × Country — {season}")
        ax.set_xlabel("Country")
        ax.set_ylabel("Model")
        plt.tight_layout()
        out = BENCHMARK_DIR / f"heatmap_mean_{metric}_{season}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outs.append(out)
    return outs


def merge_metrics_emissions(dfm: pd.DataFrame, dfe: pd.DataFrame) -> pd.DataFrame:
    # Join on model,country,season,run
    if dfm.empty or dfe.empty:
        return pd.DataFrame()
    keys = ["model", "country", "season", "run"]
    return pd.merge(dfm, dfe[keys + ["emissions_kg", "energy_kwh", "duration_sec"]], on=keys, how="left")


def plot_tradeoff_scatter(dfme: pd.DataFrame, metric: str) -> Path:
    d = dfme.dropna(subset=[metric, "emissions_kg"])
    if d.empty:
        return BENCHMARK_DIR / f"tradeoff_{metric}_vs_emissions_EMPTY.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        cmap = plt.colormaps.get_cmap('tab10')
    except Exception:
        cmap = plt.cm.get_cmap('tab10')
    countries = sorted(d["country"].unique().tolist())
    for i, country in enumerate(countries):
        sub = d[d["country"] == country]
        ax.scatter(sub["emissions_kg"].values, sub[metric].values, s=24, alpha=0.7, color=cmap(i % (getattr(cmap, 'N', 10) or 10)), label=country)
    ax.set_xlabel("Emissions (kg CO₂e)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Trade-off: {metric.upper()} vs Emissions")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out = BENCHMARK_DIR / f"tradeoff_{metric}_vs_emissions.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_top_models_bar(dfm: pd.DataFrame, metric: str, top_n: int = 10) -> Path:
    d = dfm.dropna(subset=[metric])
    if d.empty:
        return BENCHMARK_DIR / f"top{top_n}_{metric}_EMPTY.png"
    grp = d.groupby("model")[metric].mean().sort_values(ascending=True).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * top_n)))
    ax.barh(grp.index.tolist(), grp.values.tolist(), color="#2ca02c")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Top {top_n} models by mean {metric.upper()} (lower is better)")
    plt.tight_layout()
    out = BENCHMARK_DIR / f"top{top_n}_{metric}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    dfe = collect_emissions()
    if dfe.empty:
        print("No emissions files found under Results/emissions.")
    else:
        out_csv = save_aggregated(dfe)
        print(f"Aggregated CSV: {out_csv}")
        p1 = plot_box_all_models(dfe)
        print(f"Saved: {p1}")
        for p in plot_heatmaps(dfe):
            print(f"Saved: {p}")
        for p in plot_per_country_box(dfe):
            print(f"Saved: {p}")

    # Metrics-based comparisons
    dfm = collect_metrics_from_reports()
    if dfm.empty:
        print("No metrics reports found under Results/reports.")
        return 0
    outm = save_metrics_aggregated(dfm)
    print(f"Metrics CSV: {outm}")
    # Box plots for MAE and RMSE
    print(f"Saved: {plot_metric_box_all_models(dfm, 'mae')}")
    print(f"Saved: {plot_metric_box_all_models(dfm, 'rmse')}")
    # Heatmaps by season
    for p in plot_metric_heatmaps(dfm, 'mae'):
        print(f"Saved: {p}")
    for p in plot_metric_heatmaps(dfm, 'rmse'):
        print(f"Saved: {p}")
    # Trade-off plots vs emissions
    dfme = merge_metrics_emissions(dfm, dfe if 'dfe' in locals() else pd.DataFrame())
    if not dfme.empty:
        print(f"Saved: {plot_tradeoff_scatter(dfme, 'mae')}")
        print(f"Saved: {plot_tradeoff_scatter(dfme, 'rmse')}")
    # Top models
    print(f"Saved: {plot_top_models_bar(dfm, 'mae')}")
    print(f"Saved: {plot_top_models_bar(dfm, 'rmse')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
