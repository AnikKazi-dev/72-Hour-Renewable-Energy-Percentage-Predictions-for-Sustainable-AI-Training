from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
EMISSIONS_DIR = BASE / 'Results' / 'emissions'
OUT_DIR = BASE / 'Results' / 'Benchmark'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_all() -> pd.DataFrame:
    rows = []
    if not EMISSIONS_DIR.exists():
        return pd.DataFrame(rows)
    # Treat every first-level directory under emissions as a model variant
    for model_dir in sorted([p for p in EMISSIONS_DIR.iterdir() if p.is_dir()]):
        model = model_dir.name
        for country_dir in model_dir.iterdir():
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
                    # case-insensitive find of emissions column
                    emissions_col = None
                    for c in df.columns:
                        if c.lower() == 'emissions':
                            emissions_col = c
                            break
                    if emissions_col is None:
                        continue
                    vals = pd.to_numeric(df[emissions_col], errors='coerce').dropna().values
                    if vals.size == 0:
                        continue
                    # If looks cumulative/monotone use max, else sum
                    total = float(vals.max() if (vals.size == 1 or (vals[1:] >= vals[:-1]).all()) else vals.sum())
                    rows.append({
                        'model': model,
                        'country': country_dir.name,
                        'season': season_dir.name,
                        'emissions_kg': total,
                        'path': str(csv)
                    })
    return pd.DataFrame(rows)


def plot_mean_bar(df: pd.DataFrame):
    if df.empty:
        print('No emissions data found.')
        return None
    g = df.groupby('model', as_index=False)['emissions_kg'].mean()
    g = g.sort_values('emissions_kg', ascending=True)
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(g['model'], g['emissions_kg'])
    ax.set_title('Mean emissions per training — All countries & seasons (lower is better)')
    ax.set_xlabel('Model')
    ax.set_ylabel('kg CO2e (mean)')
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()
    out = OUT_DIR / 'mean_emissions_all_countries_seasons.png'
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_box_all(df: pd.DataFrame):
    if df.empty:
        return None
    order = sorted(df['model'].unique())
    data = [df.loc[df['model'] == m, 'emissions_kg'].values for m in order]
    n = len(order)
    width = max(10, min(16, 0.32 * n))  # compress width
    fig, ax = plt.subplots(figsize=(width, 4.0))
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_title('Emissions per training — All countries & seasons')
    ax.set_xlabel('Model')
    ax.set_ylabel('Emissions (kg CO2e)')
    ax.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=90, ha='center')
    plt.margins(x=0.005)
    plt.tight_layout()
    out = OUT_DIR / 'boxplot_emissions_all_countries_seasons.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_box_all_log(df: pd.DataFrame):
    if df.empty:
        return None
    order = sorted(df['model'].unique())
    data = [df.loc[df['model'] == m, 'emissions_kg'].values for m in order]
    n = len(order)
    width = max(10, min(16, 0.32 * n))
    fig, ax = plt.subplots(figsize=(width, 4.0))
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_yscale('log')
    ax.set_title('Emissions per training — All countries & seasons (log scale)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Emissions (kg CO2e, log)')
    ax.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=90, ha='center')
    plt.margins(x=0.005)
    plt.tight_layout()
    out = OUT_DIR / 'boxplot_emissions_all_countries_seasons_log.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_violin(df: pd.DataFrame):
    if df.empty:
        return None
    order = sorted(df['model'].unique())
    data = [df.loc[df['model'] == m, 'emissions_kg'].values for m in order]
    n = len(order)
    width = max(10, min(16, 0.32 * n))
    fig, ax = plt.subplots(figsize=(width, 4.0))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for b in parts['bodies']:
        b.set_alpha(0.6)
    ax.set_title('Emissions distribution (Violin) — All countries & seasons')
    ax.set_ylabel('Emissions (kg CO2e)')
    ax.set_xticks(range(1, len(order)+1))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    plt.tight_layout()
    out = OUT_DIR / 'violin_emissions_all_countries_seasons.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_strip(df: pd.DataFrame):
    if df.empty:
        return None
    order = sorted(df['model'].unique())
    n = len(order)
    width = max(10, min(16, 0.32 * n))
    fig, ax = plt.subplots(figsize=(width, 4.0))
    # jittered scatter per model
    for i, m in enumerate(order, start=1):
        y = df.loc[df['model'] == m, 'emissions_kg'].values
        if y.size == 0:
            continue
        x = np.random.normal(loc=i, scale=0.06, size=y.size)
        ax.scatter(x, y, s=8, alpha=0.6)
    ax.set_title('Emissions per training — jittered points (All countries & seasons)')
    ax.set_ylabel('Emissions (kg CO2e)')
    ax.set_xticks(range(1, len(order)+1))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    plt.tight_layout()
    out = OUT_DIR / 'strip_emissions_all_countries_seasons.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_median_iqr_bars(df: pd.DataFrame):
    if df.empty:
        return None
    stats = df.groupby('model')['emissions_kg'].agg(['median', 'quantile'])
    # quantile() alone isn't enough; compute q1,q3 explicitly
    grp = df.groupby('model')['emissions_kg']
    med = grp.median()
    q1 = grp.quantile(0.25)
    q3 = grp.quantile(0.75)
    order = med.sort_values().index.tolist()
    med = med[order]
    q1 = q1[order]
    q3 = q3[order]
    iqr_low = med - q1
    iqr_high = q3 - med
    n = len(order)
    width = max(10, min(16, 0.32 * n))
    fig, ax = plt.subplots(figsize=(width, 4.0))
    ax.bar(range(len(order)), med.values, yerr=[iqr_low.values, iqr_high.values], capsize=2)
    ax.set_title('Median emissions with IQR — All countries & seasons')
    ax.set_ylabel('Emissions (kg CO2e)')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    plt.tight_layout()
    out = OUT_DIR / 'median_iqr_emissions_all_countries_seasons.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_hist_of_means(df: pd.DataFrame):
    if df.empty:
        return None
    means = df.groupby('model')['emissions_kg'].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(means.values, bins=min(20, max(5, len(means)//2)), alpha=0.8)
    ax.set_title('Distribution of mean emissions across models')
    ax.set_xlabel('Mean emissions (kg CO2e)')
    ax.set_ylabel('Count of models')
    plt.tight_layout()
    out = OUT_DIR / 'hist_mean_emissions_models.png'
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def export_stats_csv(df: pd.DataFrame):
    if df.empty:
        return None
    agg = df.groupby('model')['emissions_kg'].agg(['count', 'mean', 'median', 'std', 'min', 'max',
                                                   lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    agg = agg.rename(columns={'<lambda_0>': 'q25', '<lambda_1>': 'q75'})
    out = OUT_DIR / 'emissions_stats_all_countries_seasons.csv'
    agg.to_csv(out)
    return out


def main():
    df = collect_all()
    b = plot_mean_bar(df)
    bx = plot_box_all(df)
    bxl = plot_box_all_log(df)
    vio = plot_violin(df)
    strip = plot_strip(df)
    med = plot_median_iqr_bars(df)
    hist = plot_hist_of_means(df)
    csv = export_stats_csv(df)
    print('Generated:', b)
    print('Generated:', bx)
    print('Generated:', bxl)
    print('Generated:', vio)
    print('Generated:', strip)
    print('Generated:', med)
    print('Generated:', hist)
    print('Generated:', csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
