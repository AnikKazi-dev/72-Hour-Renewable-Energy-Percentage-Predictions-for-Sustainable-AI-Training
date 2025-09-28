from __future__ import annotations

"""Generate LaTeX table fragments used by the paper.

Outputs written to Paper/generated/ :
  mae_summary.tex
  low_carbon_leaders.tex
  top10_mae_by_slice.tex
  top10_mae_by_slice_tabular.tex
  pareto_leaders.tex

Each data row ends with a double backslash (\\\\)."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "Results" / "Benchmark"
METRICS = BENCH / "metrics_aggregated.csv"
EMISSIONS = BENCH / "emissions_aggregated.csv"
OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(parents=True, exist_ok=True)


def latex_escape(val: object) -> str:
    if not isinstance(val, str):
        val = str(val)
    return (
        val.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def fmt(x, p: int = 3) -> str:
    try:
        return f"{float(x):.{p}f}"
    except Exception:
        return "--"


def write(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if not METRICS.exists():
    raise SystemExit(f"Missing metrics file: {METRICS}")

df = pd.read_csv(METRICS)
for c in ["country", "season", "model"]:
    if c in df.columns:
        df[c] = df[c].astype(str)
if "season" in df.columns:
    df["season"] = df["season"].str.lower()

# MAE summary
rows = []
keys = [k for k in ["country", "season"] if k in df.columns]
for (country, season), g in df.groupby(keys):
    g2 = g.dropna(subset=["mae"])
    if g2.empty:
        continue
    best = g2.loc[g2["mae"].idxmin()]
    worst = g2.loc[g2["mae"].idxmax()]
    rows.append(
        {
            "slice": f"{country} {str(season).capitalize()}",
            "best_model": latex_escape(best["model"]),
            "best_mae": best["mae"],
            "worst_model": latex_escape(worst["model"]),
            "worst_mae": worst["mae"],
            "N": len(g2),
        }
    )
mae = pd.DataFrame(rows).sort_values("slice")
lines = [
    r"\begin{tabular}{l l r l r r}",
    r"\toprule",
    r"Slice & Best (model) & Best MAE & Worst (model) & Worst MAE & N \\",
    r"\midrule",
]
for _, r in mae.iterrows():
    # End each row with double backslash for proper LaTeX line termination
    lines.append(
        f"{r['slice']} & {r['best_model']} & {fmt(r['best_mae'])} & {r['worst_model']} & {fmt(r['worst_mae'])} & {int(r['N'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write(OUT / "mae_summary.tex", lines)

# Low-carbon leaders (<=0.10 kg)
lc_path = OUT / "low_carbon_leaders.tex"
try:
    if not EMISSIONS.exists():
        raise FileNotFoundError("Missing emissions file")
    dfe = pd.read_csv(EMISSIONS)
    for c in ["country", "season", "model"]:
        if c in dfe.columns:
            dfe[c] = dfe[c].astype(str)
    if "season" in dfe.columns:
        dfe["season"] = dfe["season"].str.lower()
    join = [k for k in ["model", "country", "season", "run"] if k in df.columns and k in dfe.columns]
    if not join:
        raise ValueError("No join keys")
    merged = pd.merge(df, dfe, on=join, how="inner")
    if "emissions_kg" not in merged.columns:
        raise ValueError("emissions_kg missing")
    filt = merged[(merged["emissions_kg"] <= 0.10) & merged["mae"].notna()]
    rows = []
    for (country, season), g in filt.groupby(["country", "season"]):
        best = g.loc[g["mae"].idxmin()]
        rows.append(
            {
                "slice": f"{country} {str(season).capitalize()}",
                "model": latex_escape(best["model"]),
                "mae": best["mae"],
                "em": best["emissions_kg"],
            }
        )
    lcd = pd.DataFrame(rows).sort_values("slice")
    lines = [
        r"\begin{tabular}{l l r r}",
        r"\toprule",
        r"Slice & Model & MAE & Emissions (kg) \\",
        r"\midrule",
    ]
    for _, r in lcd.iterrows():
        lines.append(f"{r['slice']} & {r['model']} & {fmt(r['mae'])} & {fmt(r['em'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(lc_path, lines)
except Exception as e:  # pragma: no cover
    write(lc_path, [r"\begin{tabular}{l}", f"Low-carbon failed: {latex_escape(e)}", r"\end{tabular}"])

# Top-10 per slice
top_long = OUT / "top10_mae_by_slice.tex"
top_tab = OUT / "top10_mae_by_slice_tabular.tex"
try:
    by_model = df.dropna(subset=["mae"]).groupby(["country", "season", "model"], as_index=False)["mae"].mean()
    rows = []
    for (country, season), g in by_model.groupby(["country", "season"]):
        for rank, (_, r) in enumerate(g.sort_values("mae").head(10).iterrows(), start=1):
            rows.append(
                {
                    "slice": f"{country} {str(season).capitalize()}",
                    "rank": rank,
                    "model": latex_escape(r["model"]),
                    "mae": fmt(r["mae"]),
                }
            )
    tdf = pd.DataFrame(rows)
    lines_long = [
        r"\begin{longtable}{l r l r}",
        r"\caption{Top-10 MAE models per slice (averaged over runs).}\\",
        r"\toprule",
        r"Slice & Rank & Model & MAE \\",
        r"\midrule",
    ]
    for _, r in tdf.sort_values(["slice", "rank"]).iterrows():
        lines_long.append(f"{r['slice']} & {int(r['rank'])} & {r['model']} & {r['mae']} \\\\")
    lines_long += [r"\bottomrule", r"\end{longtable}"]
    write(top_long, lines_long)

    lines_tab = [
        r"\begin{tabular}{l r l r}",
        r"\toprule",
        r"Slice & Rank & Model & MAE \\",
        r"\midrule",
    ]
    for _, r in tdf.sort_values(["slice", "rank"]).iterrows():
        lines_tab.append(f"{r['slice']} & {int(r['rank'])} & {r['model']} & {r['mae']} \\\\")
    lines_tab += [r"\bottomrule", r"\end{tabular}"]
    write(top_tab, lines_tab)
except Exception as e:  # pragma: no cover
    write(top_long, [r"\begin{tabular}{l}", f"Top-10 failed: {latex_escape(e)}", r"\end{tabular}"])

# Pareto leaders
pareto_path = OUT / "pareto_leaders.tex"
try:
    if not EMISSIONS.exists():
        raise FileNotFoundError("Missing emissions file")
    dfe = pd.read_csv(EMISSIONS)
    for c in ["country", "season", "model"]:
        if c in dfe.columns:
            dfe[c] = dfe[c].astype(str)
    if "season" in dfe.columns:
        dfe["season"] = dfe["season"].str.lower()
    m_agg = df.dropna(subset=["mae"]).groupby(["country", "season", "model"], as_index=False)["mae"].mean()
    e_agg = dfe.groupby(["country", "season", "model"], as_index=False)["emissions_kg"].mean()
    agg = pd.merge(m_agg, e_agg, on=["country", "season", "model"], how="inner")

    def pareto_front(s: pd.DataFrame) -> pd.DataFrame:
        pts = s.sort_values(["mae", "emissions_kg"]).reset_index(drop=True)
        front = []
        best_em = float("inf")
        for _, row in pts.iterrows():
            if row["emissions_kg"] < best_em - 1e-12:
                front.append(row)
                best_em = row["emissions_kg"]
        return pd.DataFrame(front)

    rows = []
    for (country, season), g in agg.groupby(["country", "season"]):
        pf = pareto_front(g).nsmallest(3, ["mae", "emissions_kg"])
        for _, r in pf.iterrows():
            rows.append(
                {
                    "slice": f"{country} {str(season).capitalize()}",
                    "model": latex_escape(r["model"]),
                    "mae": fmt(r["mae"]),
                    "em": fmt(r["emissions_kg"]),
                }
            )
    pdf = pd.DataFrame(rows)
    lines = [
        r"\begin{tabular}{l l r r}",
        r"\toprule",
        r"Slice & Model & MAE & Emissions (kg) \\",
        r"\midrule",
    ]
    for _, r in pdf.sort_values(["slice", "mae", "em"]).iterrows():
        lines.append(f"{r['slice']} & {r['model']} & {r['mae']} & {r['em']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(pareto_path, lines)
except Exception as e:  # pragma: no cover
    write(pareto_path, [r"\begin{tabular}{l}", f"Pareto failed: {latex_escape(e)}", r"\end{tabular}"])

print("Tables written to", OUT)
