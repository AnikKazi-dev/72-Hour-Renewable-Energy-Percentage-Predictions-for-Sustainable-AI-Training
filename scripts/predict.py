from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Optional TF import only when using TF SavedModels
try:
    import tensorflow as tf  # noqa: F401
except Exception:
    tf = None  # type: ignore
# Robust import of countries helper whether run as package/module or script
try:
    from scripts.countries import name_for as country_name_for  # type: ignore
except Exception:
    try:
        from countries import name_for as country_name_for  # type: ignore
    except Exception:
        import sys as _sys
        _sys.path.append(str(Path(__file__).resolve().parent))
        try:
            from countries import name_for as country_name_for  # type: ignore
        except Exception as _e:
            raise ImportError("Failed to import countries.name_for") from _e

ROOT = Path(__file__).resolve().parents[1]
# Primary weights directory (new name) with backward-compatible fallback
WEIGHTS_ROOT_PRIMARY = ROOT / "Model Weights"
WEIGHTS_ROOT_LEGACY = ROOT / "ModelWeights"
WEIGHTS_ROOT = WEIGHTS_ROOT_PRIMARY
DATA_DIR = ROOT / "Data"

# Countries map can be expanded; default DE/NL example
DEFAULT_COUNTRIES = [
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

DEFAULT_MODELS = [
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
     "Cycle_LSTM_Model_v2.py",
    # "Cycle_LSTM_Model_v3.py",
     "DLinear_Model.py",
     "DLinear_Model_v2.py",
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

# Choose weights season by commenting out one; used only to locate weights on disk
DEFAULT_SEASONS = [
    "summer",
    "winter",
]

HISTORY_HOURS = 72
FORECAST_HOURS = 72
WINDOW = 72
HORIZON = 72


def _read_entsoe_token() -> str | None:
    """Read ENTSOE token from .codegreencore.config or environment variables.

    Mirrors logic in Fetch data/fetch_data.py.
    """
    import configparser
    cfg = configparser.ConfigParser()
    candidates = [
        str(ROOT / "Fetch data" / ".codegreencore.config"),
        str(ROOT / ".codegreencore.config"),
        str(Path.home() / ".codegreencore.config"),
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
    # Env fallbacks (support multiple names)
    for key in ("ENTSOE_token", "ENTSOE_TOKEN", "CODEGREEN_ENTSOE_TOKEN", "ENTSOE_API_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    return None


def fetch_recent_data_entsoe(country_code: str, hours: int = HISTORY_HOURS, no_fallback: bool = False) -> tuple[pd.DataFrame, str]:
    """Fetch recent renewable percentage using ENTSO-E API; CSV fallback if unavailable.

    Requires env ENTSOE_API_TOKEN for live queries. Computes renewable_percentage from
    generation mix as sum(renewable sources)/sum(all sources) * 100.
    """
    token = _read_entsoe_token()
    try:
        from entsoe import EntsoePandasClient  # type: ignore
        if not token:
            raise RuntimeError("Missing ENTSOE_API_TOKEN")
        client = EntsoePandasClient(api_key=token)
        end_utc = pd.Timestamp.utcnow().tz_localize("UTC")
        start_utc = end_utc - pd.Timedelta(hours=hours)
        cc = country_code.upper()
        # Prefer per-type to compute renewable share
        gen = None
        if hasattr(client, 'query_generation_per_type'):
            gen = client.query_generation_per_type(cc, start=start_utc, end=end_utc)
        if gen is None:
            gen = client.query_generation(cc, start=start_utc, end=end_utc, psr_type=None)
        # Normalize columns and compute renewables
        if isinstance(gen.columns, pd.MultiIndex):
            gen.columns = [" ".join([str(x) for x in tup if str(x) != 'nan']).strip() for tup in gen.columns]
        else:
            gen.columns = [str(c) for c in gen.columns]
        gen = gen.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        cols = {c.lower(): c for c in gen.columns}
        ren_keys = [
            'biomass',
            'hydro run-of-river and poundage',
            'hydro water reservoir',
            'hydro pumped storage',
            'geothermal',
            'marine',
            'other renewable',
            'solar',
            'wind onshore',
            'wind offshore',
        ]
        ren_cols = [cols[k] for k in ren_keys if k in cols]
        # Exclude obvious non-renewables present in some names
        ren_cols = [c for c in ren_cols if not any(bad in c.lower() for bad in ['fossil', 'coal', 'gas', 'oil', 'nuclear'])]
        total = gen.sum(axis=1)
        ren = gen[ren_cols].sum(axis=1) if ren_cols else pd.Series(0.0, index=gen.index)
        pct = (ren / total.replace(0, np.nan) * 100.0).fillna(0.0).clip(0.0, 100.0)
        out = pd.DataFrame({
            "timestamp": pct.index.tz_convert("UTC").astype(str) if getattr(pct.index, 'tz', None) is not None else pct.index.tz_localize('UTC').astype(str),
            "renewable_percentage": pct.values,
        })
        return out.tail(hours).reset_index(drop=True), "ENTSO-E API"
    except Exception as e:
        # Fallback: local CSV slice
        if no_fallback:
            raise
        season = (os.environ.get("BIONET_SEASON") or "summer").lower()
        fname = f"energy_data_{country_code.upper()}_5years_{season}.csv"
        fpath = DATA_DIR / fname
        if not fpath.exists():
            alt = f"energy_data_{country_code.upper()}_5years_{'winter' if season=='summer' else 'summer'}.csv"
            fpath = DATA_DIR / alt
        if not fpath.exists():
            raise FileNotFoundError(f"ENTSO-E fetch failed ({e}); no CSV for {country_code}")
        df = pd.read_csv(fpath)
        return df.tail(hours).reset_index(drop=True), "CSV fallback"


def build_input_window(series: np.ndarray, window: int = WINDOW) -> np.ndarray:
    if len(series) < window:
        # pad at the start with the first value
        pad = np.full((window - len(series),), series[0])
        series = np.concatenate([pad, series])
    x = series[-window:]
    # Models often expect shape (1, window, 1)
    return x.reshape(1, window, 1)


def _load_savedmodel_with_tfsmlayer(path: Path):
    # Prefer keras.layers.TFSMLayer for Keras 3 SavedModel
    try:
        from keras.layers import TFSMLayer  # type: ignore
        inp = tf.keras.Input(shape=(WINDOW, 1))
        try:
            layer = TFSMLayer(str(path), call_endpoint='serve')
        except Exception:
            layer = TFSMLayer(str(path), call_endpoint='serving_default')
        out = layer(inp)
        return tf.keras.Model(inp, out)
    except Exception as e:
        # Fallback: try a generic tf.saved_model.load wrapper
        try:
            return _load_savedmodel_generic(path)
        except Exception as e2:
            raise RuntimeError(f"Failed to load SavedModel (TFSMLayer+generic): {e} | {e2}")


def _load_savedmodel_generic(path: Path):
    if tf is None:
        raise RuntimeError("TensorFlow not available to load SavedModel")
    obj = tf.saved_model.load(str(path))
    # Try common signatures (serve/serving_default) from the signature map
    sig = None
    sig_map = getattr(obj, 'signatures', None)
    try:
        # Signature map can be a dict-like object
        if sig_map:
            if 'serve' in sig_map:
                sig = sig_map['serve']
            elif 'serving_default' in sig_map:
                sig = sig_map['serving_default']
            else:
                # pick any available signature
                try:
                    sig = next(iter(sig_map.values()))
                except Exception:
                    sig = None
    except Exception:
        sig = None
    if sig is None:
        # Some SavedModels are callable directly
        if callable(obj):
            def call_fn(x):
                return obj(x)
        else:
            raise RuntimeError("No callable signature found in SavedModel")
    else:
        # Determine input argument name(s) and build a proper call
        try:
            # structured_input_signature: (args, kwargs)
            _, kwargs_spec = sig.structured_input_signature
            input_names = list(kwargs_spec.keys()) if isinstance(kwargs_spec, dict) else []
            # Try to infer expected input rank from first kwarg spec
            first_spec = None
            if isinstance(kwargs_spec, dict) and input_names:
                first_spec = kwargs_spec[input_names[0]]
            expected_rank = len(getattr(first_spec, 'shape', []) or []) if first_spec is not None else None
        except Exception:
            input_names = []
            expected_rank = None

        def _prepare_tensor(x):
            import numpy as _np
            arr = x
            # Ensure numpy array
            try:
                if hasattr(arr, 'numpy'):
                    arr = arr.numpy()
            except Exception:
                pass
            arr = _np.asarray(arr, dtype=_np.float32)
            # If signature expects (None, 72), squeeze last dim when it's 1
            if expected_rank == 2:
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr.reshape(arr.shape[0], arr.shape[1])
                elif arr.ndim == 1:
                    arr = arr.reshape(1, -1)
            # If signature expects (None, 72, 1), ensure 3D
            elif expected_rank == 3:
                if arr.ndim == 2:
                    arr = arr[..., _np.newaxis]
                elif arr.ndim == 1:
                    arr = arr.reshape(1, -1, 1)
            return tf.convert_to_tensor(arr, dtype=tf.float32)

        def call_fn(x):
            tensor_x = _prepare_tensor(x)
            if input_names:
                # Common Keras export uses 'args_0'
                if 'args_0' in input_names:
                    return sig(args_0=tensor_x)
                # Fallback to first kw name
                return sig(**{input_names[0]: tensor_x})
            # Try positional call (may fail if signature enforces kwargs)
            return sig(tensor_x)

    class WrappedModel:
        def predict(self, x):
            out = call_fn(x)
            if isinstance(out, dict):
                out = next(iter(out.values()))
            if hasattr(out, 'numpy'):
                return out.numpy()
            # Ensure numpy array
            return np.array(out)

    return WrappedModel()


def load_saved_model(model_name: str, country: str, season: str, run_name: str | None = None) -> tuple[object, Path]:
    # Support both full country name and ISO code, and both weights roots
    cc = (country or "").upper()
    cname = country_name_for(cc) if cc else country
    base_candidates = []
    for root in (WEIGHTS_ROOT_PRIMARY, WEIGHTS_ROOT_LEGACY):
        base_candidates.extend([
            root / model_name / (cname or cc) / season.lower(),
            root / model_name / cc / season.lower(),
        ])
    base = next((b for b in base_candidates if b.exists()), None)
    if base is None:
        raise FileNotFoundError(
            "No saved weights found. Checked: "
            + ", ".join(str(b) for b in base_candidates)
        )
    # If a specific run is given, check .keras first, then directory SavedModel, then .h5
    def try_load(target: Path):
        # Try .keras
        keras_file = target if target.suffix == ".keras" else target.with_suffix(".keras") if target.is_file() else None
        if keras_file and keras_file.exists():
            return tf.keras.models.load_model(str(keras_file)), keras_file
        # Try .h5
        h5 = target.with_suffix(".h5") if not target.suffix else None
        if h5 and h5.exists():
            return tf.keras.models.load_model(str(h5)), h5
        # Try SavedModel directory
        sm_dir = target if target.is_dir() else base / target.name
        if sm_dir.is_dir() and (sm_dir / "saved_model.pb").exists():
            return _load_savedmodel_with_tfsmlayer(sm_dir), sm_dir
        # Finally, if target is a directory without saved_model.pb, try Keras loader (older TF)
        if target.is_dir():
            return tf.keras.models.load_model(str(target)), target
        raise FileNotFoundError(f"No supported model artifact at {target}")

    if run_name:
        # Exact run folder or file
        specific_dir = base / run_name
        if specific_dir.exists():
            return try_load(specific_dir)
        # Also try run_name as a file (.keras)
        specific_file = base / f"{run_name}.keras"
        if specific_file.exists():
            return try_load(specific_file)
        raise FileNotFoundError(f"Specified run '{run_name}' not found under {base}")

    # Otherwise choose most recent among directories or .keras files
    all_candidates = list(base.glob("*"))
    if not all_candidates:
        raise FileNotFoundError(f"No runs in {base}")
    all_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in all_candidates:
        try:
            return try_load(cand)
        except Exception:
            continue
    raise FileNotFoundError(f"No loadable artifacts found in {base}")


def predict_future(model, last_window: np.ndarray) -> np.ndarray:
    # Simple greedy one-shot or direct multi-step; assumes model outputs horizon=72
    y = model.predict(last_window)
    # Ensure flat array
    if y.ndim > 1:
        y = y.reshape(-1)
    return y[:FORECAST_HOURS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Predict next 72h using saved model weights and recent data")
    ap.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES), help="Comma-separated country codes")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model names (stems)")
    ap.add_argument("--season", default=None, help="Season(s) for weights lookup: summer|winter|both or comma-separated (e.g., 'summer,winter')")
    ap.add_argument("--run-name", default=None, help="Specific run folder to load within weights dir")
    ap.add_argument("--no-fallback", action="store_true", help="Fail if ENTSO-E fetch fails instead of using CSV fallback")
    args = ap.parse_args(argv)

    countries = [c.strip().upper() for c in args.countries.split(',') if c.strip()]
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    # Determine seasons to run
    if args.season:
        if args.season.lower() == "both":
            seasons = [s.lower() for s in (["summer", "winter"] if DEFAULT_SEASONS else ["summer", "winter"])]
        else:
            seasons = [s.strip().lower() for s in args.season.split(',') if s.strip()]
    elif os.environ.get("BIONET_SEASON"):
        seasons = [os.environ["BIONET_SEASON"].lower()]
    elif DEFAULT_SEASONS:
        seasons = [s.lower() for s in DEFAULT_SEASONS]
    else:
        seasons = ["summer"]

    results_all: list[dict] = []
    for season in seasons:
        # Ensure CSV fallback aligns to this season
        os.environ["BIONET_SEASON"] = season
        results: list[dict] = []
        for cc in countries:
            df, data_source = fetch_recent_data_entsoe(cc, HISTORY_HOURS, no_fallback=args.no_fallback)
            # Try to locate the main signal column
            y_col = None
            for cand in ("renewable_percentage", "value", "y", "target"):
                if cand in df.columns:
                    y_col = cand
                    break
            if y_col is None:
                # assume single-column
                y = df.iloc[:, -1].values.astype(float)
            else:
                y = df[y_col].values.astype(float)
            x = build_input_window(y, WINDOW)

            for model_name in models:
                try:
                    # Use the exact folder first (with extension), then try stem for flexibility
                    try_names = [model_name, Path(model_name).stem]
                    model = None
                    artifact = None
                    used_name = None
                    err_last = None
                    for cand in try_names:
                        try:
                            model, artifact = load_saved_model(cand, cc, season, args.run_name)
                            used_name = cand
                            break
                        except Exception as ee:  # noqa: BLE001
                            err_last = ee
                            continue
                    if model is None:
                        raise RuntimeError(err_last)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] {model_name} not found for {cc}/{season}: {e}")
                    continue
                yhat = predict_future(model, x)
                payload = {
                    "country": cc,
                    "season": season,
                    "model": used_name,
                    "forecast_hours": FORECAST_HOURS,
                    "predictions": yhat.tolist(),
                }
                results.append(payload)
                results_all.append(payload)
                # Save per-model JSON using full country name and chosen run-name (matching weights structure)
                country_full = country_name_for(cc)
                # Derive run folder name from artifact (dir name or file stem)
                run_folder = args.run_name
                if artifact is not None and not run_folder:
                    run_folder = artifact.name if artifact.is_dir() else artifact.stem
                out_dir = ROOT / "Predictions" / used_name / country_full / season / (run_folder or "latest")
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "forecast_72h.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                # Save a simple plot: last 72h context + next 72h forecast
                try:
                    ctx = y[-WINDOW:]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(range(-len(ctx), 0), ctx, label="Last 72h", color="#1f77b4")
                    ax.plot(range(0, FORECAST_HOURS), yhat, label="Next 72h (forecast)", color="#ff7f0e")
                    ax.axvline(0, color="#999", linestyle="--", linewidth=1)
                    ax.set_title(f"{used_name} — {cc} ({season}) 72h Forecast")
                    ax.set_xlabel("Hours from now")
                    ax.set_ylabel("Renewable %")
                    ax.legend(loc="best")
                    fig.tight_layout()
                    fig.savefig(out_dir / "forecast_plot.png", dpi=150)
                    plt.close(fig)
                except Exception:
                    pass
                # Save a compact textual+JSON report
                try:
                    ctx_stats = {
                        "last_value": float(ctx[-1]) if len(ctx) else None,
                        "mean": float(np.mean(ctx)) if len(ctx) else None,
                        "std": float(np.std(ctx)) if len(ctx) else None,
                        "min": float(np.min(ctx)) if len(ctx) else None,
                        "max": float(np.max(ctx)) if len(ctx) else None,
                    }
                    pred_stats = {
                        "first": float(yhat[0]) if len(yhat) else None,
                        "last": float(yhat[-1]) if len(yhat) else None,
                        "mean": float(np.mean(yhat)) if len(yhat) else None,
                        "std": float(np.std(yhat)) if len(yhat) else None,
                        "min": float(np.min(yhat)) if len(yhat) else None,
                        "max": float(np.max(yhat)) if len(yhat) else None,
                    }
                    delta_first = (pred_stats["first"] - ctx_stats["last_value"]) if (pred_stats["first"] is not None and ctx_stats["last_value"] is not None) else None
                    delta_last = (pred_stats["last"] - ctx_stats["last_value"]) if (pred_stats["last"] is not None and ctx_stats["last_value"] is not None) else None
                    report = {
                        "generated_at_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                        "country": cc,
                        "season": season,
                        "model": used_name,
                        "context_hours": WINDOW,
                        "forecast_hours": FORECAST_HOURS,
                        "context_stats": ctx_stats,
                        "forecast_stats": pred_stats,
                        "delta_first_vs_last_context": delta_first,
                        "delta_last_vs_last_context": delta_last,
                        "data_source": data_source,
                    }
                    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    summary_txt = (
                        f"Model: {used_name}\n"
                        f"Country/Season: {cc}/{season}\n"
                        f"Context last value: {ctx_stats['last_value']:.2f} | Forecast first: {pred_stats['first']:.2f} | last: {pred_stats['last']:.2f}\n"
                        f"Forecast mean±std: {pred_stats['mean']:.2f} ± {pred_stats['std']:.2f}\n"
                    )
                    (out_dir / "report.txt").write_text(summary_txt, encoding="utf-8")
                    print(summary_txt.strip())
                except Exception:
                    pass
                print(f"Saved prediction: {out_dir / 'forecast_72h.json'}")

        # aggregate results for this season
        all_out = ROOT / "Predictions" / f"predictions_{season}.json"
        all_out.parent.mkdir(parents=True, exist_ok=True)
        all_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote aggregated predictions: {all_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
