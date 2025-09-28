from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
EMISSIONS_DIR = BASE / 'Results' / 'emissions'
OUT_DIR = BASE / 'Results' / 'Benchmark'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Include DLinear, Transformer, Robust Hybrid, and Cycle LSTM variants
VARIANTS = [
    'DLinear_Model', 'DLinear_Model_v2', 'DLinear_Model_v3',
    'Transformer_Model', 'Transformer_Model_v2', 'Transformer_Model_v3',
    'Robust_Improved_Hybrid_Model', 'Robust_Improved_Hybrid_Model_v2',
    'Cycle_LSTM_Model', 'Cycle_LSTM_Model_v2', 'Cycle_LSTM_Model_v3'
]


def collect_rows():
    rows = []
    for variant in VARIANTS:
        vdir = EMISSIONS_DIR / variant
        if not vdir.exists():
            continue
        for country_dir in vdir.iterdir():
            if not country_dir.is_dir():
                continue
            for season_dir in country_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                for csv in season_dir.glob('*_emissions.csv'):
                    try:
                        df = pd.read_csv(csv)
                    except Exception:
                        continue
                    if df is None or df.empty:
                        continue
                    # find emissions column case-insensitively
                    emissions_col = None
                    for c in df.columns:
                        if c.lower() == 'emissions':
                            emissions_col = c
                            break
                    if emissions_col is None:
                        continue
                    vals = pd.to_numeric(df[emissions_col], errors='coerce').dropna().values
                    if len(vals) == 0:
                        continue
                    # heuristic: cumulative column often monotonic; else sum
                    total = float(vals.max() if (len(vals) == 1 or (vals[1:] >= vals[:-1]).all()) else vals.sum())
                    rows.append({
                        'variant': variant,
                        'country': country_dir.name,
                        'season': season_dir.name,
                        'emissions_kg': total,
                        'path': str(csv)
                    })
    return pd.DataFrame(rows)


def plot_box_log(df: pd.DataFrame, outfile: Path):
    if df.empty:
        print('No emissions data found to plot.')
        return None
    order = sorted(df['variant'].unique())
    data = [df.loc[df['variant'] == v, 'emissions_kg'].values for v in order]
    fig, ax = plt.subplots(figsize=(max(14, 0.9*len(order)), 6))
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_yscale('log')
    ax.set_title('Emissions per Run (Log Scale)')
    ax.set_ylabel('kg CO2e (log)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(outfile, dpi=170)
    plt.close(fig)
    return outfile


def main():
    df = collect_rows()
    out = plot_box_log(df, OUT_DIR / 'combined_emissions_boxplot_log.png')
    if out:
        print('Generated:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
