from __future__ import annotations
import argparse
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from scripts.countries import name_for as country_name_for

ROOT = Path(__file__).resolve().parent


# Explicit common model list (comment out items to skip). If a list is empty, we auto-discover all Models/*.py
COMMON_MODELS = [
    # "Autoformer_Model.py",
    # "Autoformer_Model_v2.py",
    # "Autoformer_Model_v3.py",
    # "CarbonCast_Model.py",
    # "CarbonCast_Model_v2.py",
    # "CarbonCast_Model_v3.py",
    # "CNN_LSTM_Model.py",
    # "CNN_LSTM_Model_v2.py",
    # "CNN_LSTM_Model_v3.py",
    # "Cycle_LSTM_Model.py",
    # "Cycle_LSTM_Model_v2.py",
    # "Cycle_LSTM_Model_v3.py",
    # "DLinear_Model.py",
    # "DLinear_Model_v2.py",
    # "DLinear_Model_v3.py",
    # "EnsembleCI_Model.py",
    # "EnsembleCI_Model_v2.py", 
    # "EnsembleCI_Model_v3.py",
    # "Hybrid_CNN_CycleLSTM_Attention_Model.py",
    # "Hybrid_CNN_CycleLSTM_Attention_Model_v2.py",
    # "Hybrid_CNN_CycleLSTM_Attention_Model_v3.py",
    # "Informer_Model.py",
    # "Informer_Model_v2.py",
    # "Informer_Model_v3.py",
    # "Mamba_Model.py",
    # "Mamba_Model_v2.py",
    # "Mamba_Model_v3.py",
    # "N_Beats_Model.py",
    # "N_Beats_Model_v2.py",
    # "N_Beats_Model_v3.py",
    # "PatchTST_Model.py",
    # "PatchTST_Model_v2.py",
    # "PatchTST_Model_v3.py",
    # "Robust_Hybrid_Model.py",
     "Robust_Improved_Hybrid_Model.py",
     "Robust_Improved_Hybrid_Model_v2.py",
    # "Temporal_Fusion_Transformer_Model.py",
    # "Temporal_Fusion_Transformer_Model_v2.py",
    # "Temporal_Fusion_Transformer_Model_v3.py",
     "Transformer_Model.py",
    # "Transformer_Model_v2.py",
    # "Transformer_Model_v3.py",
]

# Per-season model lists (edit independently if you want to diverge; otherwise both use COMMON_MODELS)
SUMMER_MODELS: list[str] = [*COMMON_MODELS]
WINTER_MODELS: list[str] = [*COMMON_MODELS]

# Seasons toggle list — comment out what you don't want to run when --season is not provided
SEASONS = [
    "summer",
    "winter",
]

# Country codes (shared for both seasons). Comment out items to skip.
COUNTRY_CODES = [
    # "AT",
    # "BA",
    # "BE",
    # "BG",
    # "CH",
    # "CZ",
     "DE",
     "DK",
    # "EE",
     "ES",
    # "FI",
    # "FR",
    # "GR",
    # "HR",
     "HU",
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


# Console formatting for emphasis (bold + green for run names)
def _bold_green(text: str) -> str:
    try:
        # Try colorama for reliable Windows support
        from colorama import Fore, Style, init  # type: ignore
        init(autoreset=True)
        return f"{Style.BRIGHT}{Fore.GREEN}{text}{Style.RESET_ALL}"
    except Exception:
        # ANSI fallback (works on modern Windows terminals)
        return f"\033[1;32m{text}\033[0m"


def _discover_models(explicit_list: Iterable[str] | None, filter_substr: str | None) -> list[Path]:
    if explicit_list:
        base = [ROOT / "Models" / m for m in explicit_list]
    else:
        base = sorted((ROOT / "Models").glob("*.py"))
    if filter_substr:
        filtered = [m for m in base if filter_substr.lower() in m.name.lower()]
        if filtered:
            return filtered
        # Fallback: search entire Models dir if explicit list yields nothing
        all_matches = [p for p in (ROOT / "Models").glob("*.py") if filter_substr.lower() in p.name.lower()]
        return all_matches
    return base


def _run_model(model_path: Path, run_name: str, quick: bool, extra_args: list[str], timeout: int | None, country_code: str, season: str):
    cmd = [sys.executable, "-m", "scripts.run_with_emissions", str(model_path),
           "--run-name", run_name, "--season", season]
    if quick:
        cmd += ["--quick", "--max-epochs", "1"]
    cmd += extra_args
    try:
        env = os.environ.copy()
        env["BIONET_SEASON"] = season
        env["ENTRY_MODEL_NAME"] = model_path.stem
        env["COUNTRY_CODE"] = country_code.upper()
        env["WEIGHTS_DIR"] = "Model Weights"
        env["SAVE_WEIGHTS"] = "1"
        result = subprocess.run(cmd, cwd=str(ROOT), text=True, timeout=timeout, env=env)
        return result
    except subprocess.TimeoutExpired:
        print(f"Timeout after {timeout}s for {model_path.name} ({season}); terminating.", file=sys.stderr)
        return subprocess.CompletedProcess(cmd, returncode=124)


def _resolve_countries(default_list: list[str], override_arg: str | None) -> list[str]:
    if override_arg:
        return [c.strip().upper() for c in override_arg.split(',') if c.strip()]
    return [c.strip().upper() for c in default_list if c.strip()]


def _run_for_season(season: str, models_list: list[str], countries_list: list[str], args: argparse.Namespace, summary: list[dict]):
    models = _discover_models(models_list, args.filter)
    if not models:
        print(f"No models found for {season}.")
        return

    # Determine countries for this season
    if season == "summer":
        countries = _resolve_countries(countries_list, args.summer_countries or args.countries)
    else:
        countries = _resolve_countries(countries_list, args.winter_countries or args.countries)

    if not countries:
        print(f"No countries selected for {season}. Use --{season}-countries or edit lists at top of main.py.")
        return

    for cc in countries:
        cc_name = country_name_for(cc)
        for m in models:
            for i in range(args.repeat):
                rn = f"{m.stem}_{cc_name}_{season}_run{i+1}"
                print(f"\n>>> Running {m.name} ({season}, {cc_name}) as {_bold_green(rn)}")
                to = args.timeout if args.timeout and args.timeout > 0 else None
                res = _run_model(m, rn, args.quick, args.extra, to, cc, season)
                if res.returncode != 0:
                    print(f"Non-zero exit code {res.returncode} for {m.name} ({season}, {cc_name}).", file=sys.stderr)
                summary.append({
                    "model": m.name,
                    "season": season,
                    "country_code": cc,
                    "country": cc_name,
                    "run_name": rn,
                    "returncode": res.returncode,
                    "timed_out": (res.returncode == 124),
                })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Models/*.py with emissions tracking for summer and/or winter")
    ap.add_argument("--season", choices=["summer", "winter", "both"], default=None, help="Override seasons to run (default uses SEASONS list)")
    ap.add_argument("--filter", default=None, help="Substring to filter model filenames")
    ap.add_argument("--quick", action="store_true", help="Quick run (limit epochs)")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat each model N times")
    ap.add_argument("--timeout", type=int, default=0, help="Per-model timeout in seconds (0 = no timeout)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args forwarded to models", default=[])
    ap.add_argument("--countries", default=None, help="Comma-separated country codes to run (applies to the selected season(s))")
    ap.add_argument("--summer-countries", dest="summer_countries", default=None, help="Override summer countries only")
    ap.add_argument("--winter-countries", dest="winter_countries", default=None, help="Override winter countries only")
    args = ap.parse_args(argv)

    summary: list[dict] = []

    # Resolve seasons to run: CLI overrides SEASONS list; else use toggled SEASONS
    if args.season == "both":
        seasons_to_run = ["summer", "winter"]
    elif args.season in ("summer", "winter"):
        seasons_to_run = [args.season]
    else:
        seasons_to_run = [s for s in SEASONS if s in ("summer", "winter")]

    if not seasons_to_run:
        print("No seasons selected. Use --season or edit SEASONS list at top of main.py.")
        return 1

    if "summer" in seasons_to_run:
        _run_for_season("summer", SUMMER_MODELS, COUNTRY_CODES, args, summary)
    if "winter" in seasons_to_run:
        _run_for_season("winter", WINTER_MODELS, COUNTRY_CODES, args, summary)

    # Write summary
    reports_dir = ROOT / "Results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if seasons_to_run == ["summer"]:
        out = reports_dir / "summer_summary.json"
    elif seasons_to_run == ["winter"]:
        out = reports_dir / "winter_summary.json"
    else:
        out = reports_dir / "seasonal_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
