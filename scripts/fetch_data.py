#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch 5 years of hourly renewable_percentage for European countries from ENTSO-E via codegreen-core,
split by Summer/Winter months, and save CSVs in Data/ as:
  - energy_data_{CC}_5years_summer.csv
  - energy_data_{CC}_5years_winter.csv

Requirements:
- codegreen-core installed and configured with an ENTSOE_token in ~/.codegreencore.config or project root.
- pandas, numpy

Usage (PowerShell):
  python "Fetch data/fetch_data.py" --countries DE,FR --years 5 --overwrite

Notes:
- We fetch year-by-year (per calendar year, with a final partial year up to now). Missing periods are skipped.
- Timestamps are converted to UTC and saved with index label 'startTime'.
- The output schema matches existing files: startTime,renewable_percentage
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import pandas as pd

CONFIG_HINT = (
    "codegreen_core requires a .codegreencore.config file with your ENTSOE_token.\n"
    "Create it in your home folder or project root with:\n\n"
    "[codegreen]\nENTSOE_token = your-token-here\n\n"
    "Docs: https://codegreen-framework.github.io/codegreen-core/setup.html"
)

# Try importing codegreen_core
try:
    from codegreen_core.data import energy as cg_energy
except Exception:
    cg_energy = None

# entsoe-py fallback support
try:
    from entsoe import EntsoePandasClient  # type: ignore
    _HAS_ENTSOE = True
except Exception:
    EntsoePandasClient = None  # type: ignore
    _HAS_ENTSOE = False

# Project paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, "Data")
os.makedirs(DATA_DIR, exist_ok=True)

# Season months (aligned with scripts/season.py)
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]

# Default ENTSO-E supported European countries (subset from codegreen docs)
# Feel free to adjust this list or pass --countries to narrow it down.
ENTSOE_COUNTRIES = [
    "AT",
    "BA",
    #"BE",
    # "BG",
    # "CH",
    # "CZ",
    # "DE",
    # "DK",
    # "EE",
    # "ES",
    # "FI",
    # "FR",
    # "GR",
    # "HR",
    # "HU",
    # "IT",
    # "LT",
    # "LU",
    # "LV",
    # "ME",
    # "NL",
    # "NO",
    # "MK",
    # "PL",
    # "PT",
    # "RO",
    # "RS",
    # "SE",
    # "SI",
    # "SK"
]

# Optional token override from CLI
ENTSOE_TOKEN_OVERRIDE: Optional[str] = None


def _daterange_years(end: datetime, years: int) -> List[tuple[datetime, datetime]]:
    """Return list of (start, end) yearly windows for the last 'years' calendar years.
    Includes full previous years and a final partial window from Jan 1 of the current year to 'end'.
    Example (years=3, end=2025-08-27):
      [(2023-01-01, 2024-01-01), (2024-01-01, 2025-01-01), (2025-01-01, 2025-08-27)]
    """
    end_naive = _to_naive(end)
    windows: List[tuple[datetime, datetime]] = []
    # Build previous full years
    for i in range(years - 1, 0, -1):
        y_start = datetime(end_naive.year - i, 1, 1)
        y_end = datetime(end_naive.year - i + 1, 1, 1)
        if y_end > end_naive:
            y_end = end_naive
        windows.append((y_start, y_end))
    # Final partial year
    cur_year_start = datetime(end_naive.year, 1, 1)
    windows.append((cur_year_start, end_naive))
    return windows


def _to_naive(dt: datetime) -> datetime:
    """Return a timezone-naive datetime (strip tzinfo)."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def fetch_country_series(country: str, years: int = 5) -> Optional[pd.Series]:
    """Fetch last 'years' years of hourly percentRenewable for a country and return a UTC-indexed Series
    named 'renewable_percentage'. Returns None if nothing fetched. Uses codegreen_core when available,
    otherwise falls back to entsoe-py if configured.
    """
    # Use naive datetimes to avoid tz-aware comparison issues
    end = datetime.now()
    windows = _daterange_years(end, years)

    parts: List[pd.Series] = []

    if cg_energy is not None:
        for w_start, w_end in windows:
            try:
                print(f"  - Fetching {country} {w_start.date()} -> {w_end.date()} ...")
                # Pass naive datetimes; codegreen_core internally compares with datetime.now() (naive)
                data_dict = cg_energy(country, _to_naive(w_start), _to_naive(w_end), "generation")
                if not (data_dict and data_dict.get("data_available") and isinstance(data_dict.get("data"), pd.DataFrame) and not data_dict["data"].empty):
                    # Try to surface an error message if present
                    if data_dict and data_dict.get("error"):
                        print(f"    Skipped (API msg): {data_dict['error']}")
                    else:
                        print("    Skipped: no data.")
                    continue

                df = data_dict["data"].copy()
                # Parse/ensure datetime timezone
                if "startTime" not in df.columns:
                    print("    Skipped: no startTime column returned.")
                    continue
                # Match notebook behavior: parse, localize if naive to Europe/Berlin, then convert to UTC
                st = pd.to_datetime(df["startTime"], errors="coerce")
                try:
                    tz_attr = getattr(st.dt, "tz", None)
                    if tz_attr is None:
                        st = st.dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
                    else:
                        st = st.dt.tz_convert("UTC")
                except Exception:
                    # Fallback: coerce directly to UTC if localization fails
                    st = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
                df["startTime"] = st
                df = df.dropna(subset=["startTime"]).set_index("startTime").sort_index()

                # Identify percent renewable column
                col = None
                for cand in ["percentRenewable", "renewable_percentage", "renewablePercent", "percent_renewable"]:
                    if cand in df.columns:
                        col = cand
                        break
                if col is None:
                    print(f"    Skipped: no 'percentRenewable' column (available: {list(df.columns)[:6]} ...)")
                    continue

                s = df[col].astype(float).resample("1h").mean().ffill().bfill()
                s = s.rename("renewable_percentage")
                parts.append(s)
            except Exception as e:
                print(f"    Error: {e}")
                continue
    else:
        print("WARNING: codegreen_core not available. Will attempt entsoe-py fallback if configured.")

    if not parts:
        # Try fallback via entsoe-py to compute renewable percentage from generation mix
        fb = _fallback_entsoe(country, years)
        return fb

    series = pd.concat(parts).sort_index()
    # Drop duplicates keeping last
    series = series[~series.index.duplicated(keep="last")]
    # Ensure UTC
    series.index = series.index.tz_convert("UTC")
    return series


def _read_entsoe_token() -> Optional[str]:
    """Read ENTSOE token from .codegreencore.config or environment variables."""
    import configparser
    cfg = configparser.ConfigParser()
    candidates = [
    os.path.join(os.path.dirname(__file__), ".codegreencore.config"),
        os.path.join(ROOT, ".codegreencore.config"),
        os.path.join(os.path.expanduser("~"), ".codegreencore.config"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                cfg.read(path)
                if cfg.has_section("codegreen") and cfg.has_option("codegreen", "ENTSOE_token"):
                    val = cfg.get("codegreen", "ENTSOE_token").strip()
                    if val:
                        return val
            except Exception:
                pass
    # Env fallbacks
    for key in ("ENTSOE_token", "ENTSOE_TOKEN", "CODEGREEN_ENTSOE_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    return None


def _fallback_entsoe(country: str, years: int) -> Optional[pd.Series]:
    """Use entsoe-py to fetch generation mix and compute renewable_percentage as percent of total."""
    if not _HAS_ENTSOE:
        print("entsoe-py not available; cannot fallback.")
        return None
    token = ENTSOE_TOKEN_OVERRIDE or _read_entsoe_token()
    if not token:
        print("Missing ENTSOE token. Create .codegreencore.config with [codegreen]\nENTSOE_token = <your-token>\n or set ENTSOE_TOKEN env var.")
        return None
    try:
        client = EntsoePandasClient(api_key=token)
    except Exception as e:
        print(f"Failed to init Entsoe client: {e}")
        return None

    end = datetime.now()
    windows = _daterange_years(end, years)

    parts: List[pd.Series] = []
    for w_start, w_end in windows:
        try:
            start_ts = pd.Timestamp(w_start, tz='UTC')
            end_ts = pd.Timestamp(w_end, tz='UTC')
            df = None
            try:
                # entsoe-py: generation per type returns a DataFrame of technologies
                if hasattr(client, 'query_generation_per_type'):
                    df = client.query_generation_per_type(country, start=start_ts, end=end_ts)
                else:
                    df = client.query_generation(country, start=start_ts, end=end_ts)
            except Exception as e:
                print(f"    Fallback query error: {e}")
                df = None
            if df is None or df.empty:
                print("    Fallback: no generation data for window.")
                continue
            # Normalize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join([str(x) for x in tup if str(x) != 'nan']).strip() for tup in df.columns]
            else:
                df.columns = [str(c) for c in df.columns]

            # Identify renewable columns heuristically
            renewable_keywords = [
                'solar', 'wind onshore', 'wind offshore', 'hydro', 'biomass', 'geothermal', 'marine', 'other renewable'
            ]
            cols_lower = {c.lower(): c for c in df.columns}
            ren_cols = []
            for kw in renewable_keywords:
                for lc, orig in cols_lower.items():
                    if kw in lc:
                        ren_cols.append(orig)
            # Exclude obvious non-renewables
            ren_cols = sorted(set([c for c in ren_cols if not any(x in c.lower() for x in ['fossil', 'coal', 'gas', 'oil', 'nuclear'])]))
            if not ren_cols:
                print("    Fallback: no renewable columns found in generation mix.")
                continue

            df_num = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            renewable_sum = df_num[ren_cols].sum(axis=1)
            total_sum = df_num.sum(axis=1)
            pct = (renewable_sum / total_sum.replace(0, pd.NA)) * 100.0
            pct = pct.fillna(0.0).clip(lower=0.0, upper=100.0)
            pct.index = pct.index.tz_convert('UTC') if pct.index.tz is not None else pct.index.tz_localize('UTC')
            pct = pct.resample('1h').mean().ffill().bfill()
            pct.name = 'renewable_percentage'
            parts.append(pct)
        except Exception as e:
            print(f"    Fallback error: {e}")
            continue
    if not parts:
        return None
    series = pd.concat(parts).sort_index()
    series = series[~series.index.duplicated(keep='last')]
    return series


def save_country_summer_winter(country: str, series: pd.Series, years: int = 5) -> None:
    """Filter series by months and save two CSVs into Data/ matching existing schema."""
    if series.empty:
        print(f"No data to save for {country}.")
        return

    # Filter to last 'years' years explicitly
    end = series.index.max()
    start = end - pd.DateOffset(years=years)
    series = series.loc[start:end]

    df = series.to_frame()
    df_summer = df[df.index.month.isin(SUMMER_MONTHS)]
    df_winter = df[df.index.month.isin(WINTER_MONTHS)]

    summer_path = os.path.join(DATA_DIR, f"energy_data_{country}_5years_summer.csv")
    winter_path = os.path.join(DATA_DIR, f"energy_data_{country}_5years_winter.csv")

    df_summer.to_csv(summer_path, index_label="startTime")
    df_winter.to_csv(winter_path, index_label="startTime")
    print(f"  Saved: {os.path.basename(summer_path)} ({len(df_summer)})")
    print(f"  Saved: {os.path.basename(winter_path)} ({len(df_winter)})")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E renewable_percentage for European countries using codegreen-core (with entsoe-py fallback)")
    parser.add_argument("--countries", type=str, default=",".join(ENTSOE_COUNTRIES), help="Comma-separated 2-letter country codes (default: many ENTSO-E countries)")
    parser.add_argument("--years", type=int, default=5, help="Years of history to fetch (default: 5)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files; otherwise skip countries already present")
    parser.add_argument("--entsoe-token", dest="entsoe_token", type=str, default=None, help="Override ENTSOE token (optional). If omitted, read from .codegreencore.config or env")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if cg_energy is None and not _HAS_ENTSOE:
        print("WARNING: Neither codegreen_core nor entsoe-py are available. Will only skip existing files; fetching new data won't work.")
        print(CONFIG_HINT)

    global ENTSOE_TOKEN_OVERRIDE
    ENTSOE_TOKEN_OVERRIDE = args.entsoe_token.strip() if args.entsoe_token else None

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    print(f"Countries: {countries}")

    for cc in countries:
        summer_path = os.path.join(DATA_DIR, f"energy_data_{cc}_5years_summer.csv")
        winter_path = os.path.join(DATA_DIR, f"energy_data_{cc}_5years_winter.csv")
        if not args.overwrite and os.path.exists(summer_path) and os.path.exists(winter_path):
            print(f"Skipping {cc}: files already exist (use --overwrite to refetch)")
            continue

        print(f"\nFetching country: {cc}")
        series = fetch_country_series(cc, years=args.years)
        if series is None:
            print(f"No data fetched for {cc}.")
            continue
        save_country_summer_winter(cc, series, years=args.years)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
