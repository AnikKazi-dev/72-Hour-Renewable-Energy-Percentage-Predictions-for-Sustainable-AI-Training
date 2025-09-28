from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
METRICS_DIR = BASE / 'Results' / 'metrics'
OUT_DIR = BASE / 'Results' / 'Benchmark'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ['Germany', 'Denmark', 'Hungary', 'Spain']
SEASONS = ['summer', 'winter']


def safe_read_json(path: Path):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def collect_mean_mae(country: str):
    rows = []
    for model_dir in sorted(METRICS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        vals = []
        for season in SEASONS:
            p = model_dir / country / season / 'metrics.json'
            if not p.exists():
                continue
            payload = safe_read_json(p)
            if not payload or 'mae' not in payload:
                continue
            vals.append(float(payload['mae']))
        if not vals:
            continue
        mean_mae = sum(vals) / len(vals)
        rows.append({'model': model_name, 'country': country, 'mean_mae': mean_mae})
    return pd.DataFrame(rows)


def plot_country(country: str, sort_ascending: bool = True):
    df = collect_mean_mae(country)
    if df.empty:
        print(f'No metrics for {country}')
        return None
    df = df.sort_values('mean_mae', ascending=sort_ascending)
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(df['model'], df['mean_mae'])
    ax.set_title(f'MAE by model — {country} (mean of summer+winter, lower is better)')
    ax.set_xlabel('Model')
    ax.set_ylabel('MAE')
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()
    out = OUT_DIR / f'mae_by_model_{country.lower()}_mean_seasons.png'
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print('Generated:', out)
    return out


def main():
    outs = []
    for country in COUNTRIES:
        out = plot_country(country)
        if out:
            outs.append(out)
    print('Done. Files:', *outs, sep='\n  ')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
