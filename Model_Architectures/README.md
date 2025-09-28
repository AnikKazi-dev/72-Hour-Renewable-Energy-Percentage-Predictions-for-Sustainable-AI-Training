# Exported Model Architectures

This folder contains visualizations (PNG) and text summaries (TXT) for each model definition located in `Models/`.
Generation Script: `scripts/export_model_architectures.py`

## Usage

Run the exporter after activating your environment:

```
python scripts/export_model_architectures.py
```

Outputs:

- One `.txt` per model (always) with `model.summary()`
- One `.png` per model (if `pydot`/Graphviz and `plot_model` succeed)

## Fallbacks & Notes

- If a builder function cannot be detected, the script searches for a pre-built Keras Model object in the module.
- If `tensorflow` or `plot_model` dependencies are missing, only text summaries are generated (no images).
- Heuristic default shapes: look_back=72, features=1, horizon=72; adjust in the script if needed.
- Multiple builder functions in a single file produce files named `<module>__<function>.txt/png`.
- Errors during import or instantiation are logged but do not stop the batch.
