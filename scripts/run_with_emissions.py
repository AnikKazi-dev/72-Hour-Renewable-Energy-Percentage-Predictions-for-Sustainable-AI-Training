from __future__ import annotations
import argparse
import runpy
import sys
from pathlib import Path
from typing import List, Callable
import os
import re

from .emissions import emissions_tracker
from .results_manager import ResultsManager
from . import metrics as metrics_helper
from .countries import name_for as country_name_for

ROOT = Path(__file__).resolve().parents[1]


def _resolve_entry(entry: str) -> Path:
    p = ROOT / entry
    if p.suffix != ".py":
        p = ROOT / f"{entry}.py"
    if not p.exists():
        cand = ROOT / "Models" / (entry if entry.endswith(".py") else f"{entry}.py")
        if cand.exists():
            return cand
        raise FileNotFoundError(f"Entry script not found: {entry}")
    return p


def _patch_matplotlib(target_dir: Path):
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return None

    saved: list[str] = []
    orig_savefig: Callable[..., object] = getattr(plt, "savefig")

    def savefig_wrapper(fname, *args, **kwargs):  # noqa: ANN001
        nonlocal saved
        out = target_dir / (Path(fname).name if isinstance(fname, (str, Path)) else "figure.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        result = orig_savefig(str(out), *args, **kwargs)
        saved.append(str(out))
        return result

    plt.savefig = savefig_wrapper  # type: ignore[assignment]

    def restore():
        plt.savefig = orig_savefig  # type: ignore[assignment]
        return saved

    return restore


def _patch_keras_quick(max_epochs: int = 1):
    try:
        import tensorflow as tf  # noqa: WPS433
        from tensorflow.keras.models import Model as KModel  # noqa: WPS433
    except Exception:
        return None

    orig_fit = KModel.fit

    def fit_wrapper(self, *args, **kwargs):  # noqa: ANN001
        epochs = kwargs.get("epochs", max_epochs)
        kwargs["epochs"] = min(int(epochs), int(max_epochs))
        return orig_fit(self, *args, **kwargs)

    KModel.fit = fit_wrapper  # type: ignore[assignment]

    def restore():
        KModel.fit = orig_fit  # type: ignore[assignment]
        return True

    return restore


def _patch_pandas_dataset(season: str | None):
    if not season:
        return None
    season = season.lower()
    try:
        import pandas as pd  # noqa: WPS433
    except Exception:
        return None

    orig_read_csv = pd.read_csv
    data_dir = ROOT / "Data"

    def read_csv_wrapper(filepath_or_buffer, *args, **kwargs):  # noqa: ANN001
        path = filepath_or_buffer
        try:
            p = Path(filepath_or_buffer) if isinstance(filepath_or_buffer, (str, Path)) else None
        except Exception:
            p = None
        if p is not None:
            name = p.name
            if re.search(r"energy_data_.*_(winter|summer)\.csv", name, re.IGNORECASE):
                new_name = re.sub(r"(winter|summer)", season, name, flags=re.IGNORECASE)
                candidate = (p.parent / new_name) if p.is_absolute() else (data_dir / new_name)
                path = str(candidate)
            elif name.startswith("energy_data_") and not p.exists():
                candidate = (data_dir / name)
                if candidate.exists():
                    path = str(candidate)
        return orig_read_csv(path, *args, **kwargs)

    pd.read_csv = read_csv_wrapper  # type: ignore[assignment]

    def restore():
        pd.read_csv = orig_read_csv  # type: ignore[assignment]
        return True

    return restore


def run_entrypoint(entry_path: Path, script_args: List[str]):
    sys.argv = [str(entry_path)] + script_args
    # Return globals dict so we can post-process metrics if possible
    return runpy.run_path(str(entry_path))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Python entrypoint with CodeCarbon tracking")
    parser.add_argument("entry", help="Path or name of the .py under project root or Models/")
    parser.add_argument("--run-name", default=None, help="Name for results folders and emissions file")
    parser.add_argument("--no-image-redirect", action="store_true", help="Do not redirect matplotlib images to Results/")
    parser.add_argument("--quick", action="store_true", help="Quick mode: cap Keras epochs to 1 (or --max-epochs)")
    parser.add_argument("--max-epochs", type=int, default=1, help="When --quick is set, limit Keras epochs to this value")
    parser.add_argument("--season", choices=["winter", "summer"], default=None, help="Force dataset season mapping for CSV reads")
    parser.add_argument("--cwd", default=None, help="Temporarily set working directory while running entry")

    # Allow any remaining args to be forwarded to the entry script
    args, forward_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    entry_path = _resolve_entry(args.entry)
    run_name = args.run_name or Path(args.entry).stem

    # Determine model name and season for folder organization
    model_name = Path(args.entry).stem
    season_env = (args.season or (os.environ.get("BIONET_SEASON") or "unknown")).lower()
    # Initialize results manager with model/season-aware paths
    rm = ResultsManager(run_name=run_name, model_name=model_name, season=season_env)

    restore = None
    if not args.no_image_redirect:
        restore = _patch_matplotlib(rm.image_dir)

    restore_keras = None
    if args.quick:
        # Mark quick mode for models that want to adapt internally
        os.environ["BIONET_QUICK"] = "1"
        os.environ["BIONET_MAX_EPOCHS"] = str(int(args.max_epochs))
        restore_keras = _patch_keras_quick(args.max_epochs)
    # Set env so models can read selected season
    if args.season:
        os.environ["BIONET_SEASON"] = args.season
    # Propagate run name to models for nicer file naming
    os.environ["RUN_NAME"] = run_name
    # Propagate model name for emissions pathing
    os.environ["ENTRY_MODEL_NAME"] = model_name
    restore_pd = _patch_pandas_dataset(args.season)

    prev_cwd = os.getcwd()
    if args.cwd:
        os.chdir(args.cwd)
    try:
        with emissions_tracker(task_name=run_name, output_file=f"{run_name}_emissions.csv") as (tracker, emissions_csv):
            if args.season:
                print(f"[runner] Season set to: {args.season}")
            try:
                g = run_entrypoint(entry_path, forward_args)
            finally:
                saved_imgs = restore() if restore else []
                if restore_keras:
                    try:
                        restore_keras()
                    except Exception:
                        pass
                if restore_pd:
                    try:
                        restore_pd()
                    except Exception:
                        pass
                # try to compute metrics if present in globals
                metrics_payload = {}
                try:
                    y_true = None
                    y_pred = None
                    # common names across our converted notebooks
                    for k_true in ("y_test_inversed", "y_true", "y_test"):
                        if k_true in g:
                            y_true = g[k_true]
                            break
                    for k_pred in ("y_pred_inversed", "y_pred", "y_pred_scaled", "pred"):
                        if k_pred in g:
                            y_pred = g[k_pred]
                            break
                    if y_true is not None and y_pred is not None:
                        metrics_payload = metrics_helper.as_dict(y_true, y_pred)
                        rm.save_metrics(metrics_payload)
                except Exception:
                    pass

                # Optionally save trained model weights
                try:
                    save_flag = os.environ.get("SAVE_WEIGHTS", "0") == "1"
                    weights_dir = os.environ.get("WEIGHTS_DIR", "Model Weights")
                    cc = (os.environ.get("COUNTRY_CODE") or os.environ.get("BIONET_COUNTRY_CODE") or "XX").upper()
                    country = country_name_for(cc) if cc else "All"
                    season = os.environ.get("BIONET_SEASON", "unknown").lower()
                    if save_flag:
                        import tensorflow as tf  # noqa: WPS433
                        # Try common model variable names from converted notebooks
                        model_obj = None
                        for k in ("model", "model_v1", "model_v2", "model_v3",
                                  "solar_model", "best_model", "forecast_model"):
                            if k in g and hasattr(g[k], "save"):
                                model_obj = g[k]
                                break
                        # Fallback: scan all globals for a Keras model-like object
                        if model_obj is None:
                            for _name, _val in g.items():
                                try:
                                    if hasattr(_val, "save") and isinstance(_val, tf.keras.Model):
                                        model_obj = _val
                                        break
                                except Exception:
                                    continue
                        if model_obj is not None:
                            out_dir = ROOT / weights_dir / model_name / country / season
                            out_dir.mkdir(parents=True, exist_ok=True)
                            # Save SavedModel for portability
                            save_path = out_dir / f"{run_name}"
                            ok = False
                            # Keras 3 supports export() for SavedModel; prefer that when available
                            try:
                                if hasattr(model_obj, "export"):
                                    model_obj.export(str(save_path))
                                    ok = True
                                else:
                                    # Try Keras save to a directory (older TF will write SavedModel)
                                    model_obj.save(str(save_path))
                                    ok = True
                            except Exception:
                                # Try TensorFlow SavedModel as last resort
                                try:
                                    tf.saved_model.save(model_obj, str(save_path))
                                    ok = True
                                except Exception:
                                    ok = False
                            if ok:
                                print(f"[runner] Saved weights to {save_path}")
                except Exception:
                    pass

                # Append a tiny report linking emissions and images
                payload = {
                    "entry": str(entry_path.relative_to(ROOT)),
                    "images": saved_imgs,
                    "season": args.season,
                    "cwd": os.getcwd(),
                }
                if metrics_payload:
                    payload["metrics"] = metrics_payload
                if emissions_csv:
                    payload["emissions_csv"] = str(emissions_csv)
                rm.append_report(payload)
    finally:
        os.chdir(prev_cwd)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
