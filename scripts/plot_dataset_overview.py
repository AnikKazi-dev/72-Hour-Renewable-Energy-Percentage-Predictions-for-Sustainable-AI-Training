import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / 'Data'
OUT_DIR = BASE / 'Results' / 'Benchmark'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = {
    'DE': 'Germany',
    'DK': 'Denmark',
    'HU': 'Hungary',
    'ES': 'Spain',
}
SEASONS = ['summer', 'winter']
YEARS = 5
DEFAULT_DAYS = 1


def load_country_season(code: str, season: str):
    fname = DATA_DIR / f'energy_data_{code}_{YEARS}years_{season}.csv'
    if not fname.exists():
        return None
    try:
        df = pd.read_csv(fname)
    except Exception:
        return None
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    # Expect 'startTime' and 'renewable_percentage'
    st_col = None
    rp_col = None
    for c in df.columns:
        lc = c.lower()
        if 'start' in lc:
            st_col = c
        if 'renewable' in lc:
            rp_col = c
    if st_col is None or rp_col is None:
        return None
    # parse datetime and set index hourly
    df[st_col] = pd.to_datetime(df[st_col], utc=True, errors='coerce')
    df = df.dropna(subset=[st_col, rp_col]).copy()
    df = df.sort_values(st_col)
    df = df.set_index(st_col)
    # force hourly frequency to handle missing
    s = df[rp_col].astype(float).asfreq('H')
    return s


def plot_time_series(samples: dict, title: str, out_name: str, days=7):
    # Plot last N days per country-season
    fig, axes = plt.subplots(len(samples), 1, figsize=(12, 2.2*len(samples)), sharex=True)
    if len(samples) == 1:
        axes = [axes]
    for ax, (label, series) in zip(axes, samples.items()):
        if series is None or series.empty:
            ax.text(0.5, 0.5, f'No data: {label}', ha='center', va='center', transform=ax.transAxes)
            continue
        tail = series.dropna().iloc[-24*days:]
        ax.plot(tail.index, tail.values, lw=1.2)
        ax.set_ylabel('%')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (last {} days)'.format(days))
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_hourly_profile(series_map: dict, title: str, out_name: str):
    # Average profile by hour of day across the whole season
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, s in series_map.items():
        if s is None or s.empty:
            continue
        prof = s.dropna()
        if prof.empty:
            continue
        prof = prof.groupby(prof.index.hour).mean()
        ax.plot(prof.index, prof.values, marker='o', label=label)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Avg Renewable %')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title(title)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_germany_seasons(summer_s: Optional[pd.Series], winter_s: Optional[pd.Series], days: int, out_name: str):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=False, sharey=True)
    # Summer panel
    ax = axes[0]
    if summer_s is None or summer_s.dropna().empty:
        ax.text(0.5, 0.5, 'No data: Germany (Summer)', ha='center', va='center', transform=ax.transAxes)
    else:
        tail = summer_s.dropna().iloc[-24*days:]
        ax.plot(tail.index, tail.values, color='#e67e22', lw=1.4)
        ax.set_ylabel('Renewable %')
        # Date range annotation
        d_start = tail.index[0].strftime('%Y-%m-%d')
        d_end = tail.index[-1].strftime('%Y-%m-%d')
        date_part = d_start if d_start == d_end else f'{d_start} → {d_end}'
        ax.set_title(f'Germany — Summer ({days} day{"s" if days!=1 else ""})  [{date_part}]')
    if summer_s is None or summer_s.dropna().empty:
        ax.set_title('Germany — Summer (no recent data)')
    ax.grid(True, alpha=0.3)
    # Winter panel
    ax = axes[1]
    if winter_s is None or winter_s.dropna().empty:
        ax.text(0.5, 0.5, 'No data: Germany (Winter)', ha='center', va='center', transform=ax.transAxes)
    else:
        tail = winter_s.dropna().iloc[-24*days:]
        ax.plot(tail.index, tail.values, color='#2980b9', lw=1.4)
        ax.set_ylabel('Renewable %')
        d_start = tail.index[0].strftime('%Y-%m-%d')
        d_end = tail.index[-1].strftime('%Y-%m-%d')
        date_part = d_start if d_start == d_end else f'{d_start} → {d_end}'
        ax.set_title(f'Germany — Winter ({days} day{ "s" if days!=1 else ""})  [{date_part}]')
    if winter_s is None or winter_s.dropna().empty:
        ax.set_title('Germany — Winter (no recent data)')
    ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time')
    fig.suptitle('Germany Renewable Percentage — Seasonal Snapshots', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _extract_last_day_slice(
    series: Optional[pd.Series],
    start_hour: int = 6,
    end_hour: int = 24,
    lookback_days: int = 7,
    require_full: bool = True,
    interpolate_missing: bool = True,
) -> Optional[pd.Series]:
    """Return a recent day's slice between start_hour (inclusive) and end_hour (exclusive).

    Improvements:
    - Scans up to ``lookback_days`` backwards to find the most recent *complete* window
      (i.e. has all expected hours) when ``require_full`` is True.
    - If no complete window is found, falls back to the most recent *partial* day.
    - If still partial and ``interpolate_missing`` is True, reindexes to full hour grid and
      linearly interpolates gaps so the plot always shows the expected number of points.
    """
    if series is None or series.dropna().empty:
        return None
    s = series.dropna().sort_index()
    # Limit search span for efficiency
    span = s.iloc[-(lookback_days * 30):]  # heuristic slice
    if span.empty:
        return None
    expected_count = end_hour - start_hour  # e.g. 24-6 = 18
    # Collect unique dates descending
    dates = sorted({ts.date() for ts in span.index}, reverse=True)
    chosen = None
    for d in dates:
        day_slice = span[span.index.date == d]
        window = day_slice[(day_slice.index.hour >= start_hour) & (day_slice.index.hour < end_hour)]
        if not window.empty:
            if not require_full or len(window) == expected_count:
                chosen = window
                break
            # Keep candidate if we don't find a full one later
            if chosen is None:
                chosen = window  # partial placeholder
    if chosen is None or chosen.empty:
        return None
    # If we require full and didn't get full length, optionally interpolate
    if require_full and len(chosen) != expected_count and interpolate_missing:
        # Build full index for that date
        d = chosen.index[-1].date()
        # Determine timezone awareness
        last_idx = chosen.index[-1]
        if last_idx.tzinfo is not None:
            full_index = pd.date_range(
                start=pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=start_hour, tz=last_idx.tzinfo),
                end=pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=end_hour - 1, tz=last_idx.tzinfo),
                freq='H'
            )
        else:
            full_index = pd.date_range(
                start=pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=start_hour),
                end=pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=end_hour - 1),
                freq='H'
            )
        reindexed = chosen.reindex(full_index)
        # Linear interpolate internal gaps; forward/back fill edges if minimal
        reindexed = reindexed.interpolate(limit_direction='both')
        chosen = reindexed
    return chosen


def plot_germany_daytime_window(summer_s: Optional[pd.Series], winter_s: Optional[pd.Series], start_hour: int = 6, end_hour: int = 24, out_name: str = 'germany_daytime_window.png'):
    """Plot Germany summer vs winter for the most recent day from start_hour to end_hour.

    Produces a single figure with both lines (Summer, Winter) for quick diurnal comparison.
    """
    sum_slice = _extract_last_day_slice(summer_s, start_hour, end_hour, lookback_days=10, require_full=True)
    win_slice = _extract_last_day_slice(winter_s, start_hour, end_hour, lookback_days=10, require_full=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    has_any = False
    if sum_slice is not None:
        # Extract date label
        sd = sum_slice.index[0].strftime('%Y-%m-%d')
        ax.plot(sum_slice.index.hour, sum_slice.values, marker='o', label=f'Summer {sd} ({len(sum_slice)} pts)', color='#e67e22')
        has_any = True
    if win_slice is not None:
        wd = win_slice.index[0].strftime('%Y-%m-%d')
        ax.plot(win_slice.index.hour, win_slice.values, marker='o', label=f'Winter {wd} ({len(win_slice)} pts)', color='#2980b9')
        has_any = True
    if not has_any:
        ax.text(0.5, 0.5, 'No data for requested window', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlim(start_hour, end_hour - 1)
    ax.set_xticks(list(range(start_hour, end_hour, 2)))
    ax.set_xlabel('Hour of Day (UTC)')
    ax.set_ylabel('Renewable %')
    title_dates = []
    if sum_slice is not None:
        title_dates.append(sum_slice.index[0].strftime('%Y-%m-%d'))
    if win_slice is not None:
        title_dates.append(win_slice.index[0].strftime('%Y-%m-%d'))
    date_str = ' / '.join(title_dates) if title_dates else 'No Data'
    ax.set_title(f'Germany Renewable % — {start_hour:02d}:00 to {end_hour:02d}:00  [{date_str}]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main():
    data = {season: {} for season in SEASONS}
    for season in SEASONS:
        for code, name in COUNTRIES.items():
            s = load_country_season(code, season)
            label = f'{name} ({season.capitalize()})'
            data[season][label] = s

    # Time series panels (last N days) per season
    for season in SEASONS:
        days = DEFAULT_DAYS
        out = plot_time_series(
            data[season],
            title=f'Renewable % — Last {days} Days Snapshots ({season.capitalize()})',
            out_name=f'dataset_timeseries_{season}.png',
            days=days
        )
        print('Generated:', out)

    # Hourly profiles overlay (season wise)
    for season in SEASONS:
        out = plot_hourly_profile(
            data[season],
            title=f'Average Hourly Profile ({season.capitalize()})',
            out_name=f'dataset_hourly_profile_{season}.png'
        )
        print('Generated:', out)

    # Combined country comparison across seasons at once (profiles)
    combined_map = {}
    for season in SEASONS:
        for code, name in COUNTRIES.items():
            s = data[season].get(f'{name} ({season.capitalize()})')
            combined_map[f'{name} - {season.capitalize()}'] = s
    out = plot_hourly_profile(
        combined_map,
        title='Average Hourly Profiles — Countries × Seasons',
        out_name='dataset_hourly_profile_countries_seasons.png'
    )
    print('Generated:', out)

    # Germany (Summer + Winter) in one image (two stacked panels)
    de_summer = data['summer'].get('Germany (Summer)')
    de_winter = data['winter'].get('Germany (Winter)')
    out = plot_germany_seasons(
        de_summer,
        de_winter,
        days=DEFAULT_DAYS,
        out_name='dataset_germany_summer_winter_timeseries.png'
    )
    print('Generated:', out)

    # Germany daytime 06:00–24:00 window (most recent day) overlay
    out_daytime = plot_germany_daytime_window(
        de_summer,
        de_winter,
        start_hour=6,
        end_hour=24,
        out_name='dataset_germany_daytime_0600_2400.png'
    )
    print('Generated:', out_daytime)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
