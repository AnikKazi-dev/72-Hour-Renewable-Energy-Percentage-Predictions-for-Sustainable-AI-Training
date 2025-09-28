# Carbon Emission & Energy Efficiency Report

Generated from project code inspection (date: 2025-09-15). Includes factual instrumentation plus one clearly labeled hypothetical estimation example (option b as requested).

## 1. Executive Summary

This project integrates carbon accounting into the model development workflow using CodeCarbon. Emissions are tracked per model, per country, per season, and per run through a structured directory layout under `Results/emissions/`. A downstream benchmarking pipeline aggregates emissions with accuracy metrics (MAE, RMSE, etc.) enabling trade‑off visualization (performance vs. carbon footprint). The codebase already provides:

- Automated emissions tracking context manager (`scripts/emissions.py`).
- Batch collection and aggregation routines (`scripts/benchmark.py`).
- Visualization outputs (box plots, heatmaps, scatter trade-offs).
- Table generation for “low-carbon leaders” in `Paper/generate_tables.py`.

## 2. Instrumentation Stack

| Component               | File                                       | Role                                         | Key Settings                                                        |
| ----------------------- | ------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------- |
| CodeCarbon Config       | `codecarbon.config`                        | Default tracker parameters                   | `country=DEU`, `measure_power_secs=15`, output dir override by code |
| Tracker Wrapper         | `scripts/emissions.py`                     | Context manager standardizing output path    | Organizes path `<Model>/<Country>/<season>/<run>_emissions.csv`     |
| Runner Harness          | `scripts/run_with_emissions.py`            | Executes a model script inside tracker       | Injects env vars for naming consistency                             |
| Aggregation & Plots     | `scripts/benchmark.py`                     | Collects emissions + metrics, produces plots | Box/heatmap/trade-off generation                                    |
| Metrics/Emissions Merge | `merge_metrics_emissions()` (benchmark.py) | Joins accuracy & footprint                   | Enables Pareto-style analysis                                       |
| Paper Tables            | `Paper/generate_tables.py`                 | Creates LaTeX summaries                      | “Low-carbon leaders”, Pareto fronts                                 |

### 2.1 Path Convention

`Results/emissions/<ModelName>/<CountryName>/<season>/<run>_emissions.csv`

Derivation details:

- `ModelName` from env: `ENTRY_MODEL_NAME` or `RUN_MODEL_NAME`.
- `COUNTRY_CODE` mapped via `countries.name_for()` for human-readable folder.
- `season` from `BIONET_SEASON` (lowercased; fallback `unknown`).

### 2.2 Tracker Lifecycle (Simplified)

```python
with emissions_tracker(task_name=run_name, output_file=f"{run_name}_emissions.csv") as (tracker, csv_path):
    # run training / inference here
    pass  # tracker.stop() auto-invoked in context exit
```

## 3. Data Flow Overview

1. Model script executed (directly or via orchestrator `main.py`).
2. Environment variables set (model name, country, season).
3. `emissions_tracker` creates directory + starts CodeCarbon.
4. CodeCarbon periodically measures power every 15s, writing incremental CSV rows.
5. Training ends → tracker stops → final CSV contains cumulative columns (e.g., `emissions`, `energy_consumed`, `duration`).
6. `benchmark.py` scans `Results/emissions/`, aggregates totals per run.
7. Metrics JSON (if present) merged with emissions to produce holistic evaluation.
8. Plots & tables generated under `Results/Benchmark/` and `Paper/` artifacts.

## 4. Emissions CSV Semantics

Columns (observed / expected from CodeCarbon):

- `emissions` (kg CO₂e) – may be cumulative.
- `energy_consumed` or `energy_kwh` (kWh) – cumulative or instantaneous snapshots.
- `duration` or `duration_sec` – runtime seconds.
- Optional: `cpu_energy`, `gpu_energy` (if GPU present), `ram_energy` (dependent on CodeCarbon version).

Aggregation logic in `_load_emissions_total`:

- Detects cumulative vs additive by monotonicity of the column; uses `max` if non-decreasing else `sum`.
- Applies same approach to `energy_*` and `duration`.

## 5. Model Parameter Landscape (Relative Complexity)

Representative total parameter counts from exported summaries (subset for illustration):
| Model Variant | Params | Notes |
|---------------|--------|-------|
| DLinear Base / V2 | 10,512 | Ultra-light linear decomposition |
| DLinear V3 | 148,624 | MLP expansion increases energy per step |
| Transformer Base | 12,972 | Minimal attention (no embedding) |
| Transformer V2 | 573,072 | Positional embedding + 2 blocks |
| Transformer V3 | 561,296 | Deeper + regularized (slightly fewer params than V2) |
| CNN_LSTM V3 | 801,096 | Convs + stacked BiLSTM + dense head |
| N-Beats V3 | 1,538,624 | Large fully-connected stacks |
| EnsembleCI V3 | 1,360,328 | Aggregated ensemble overhead |
| Hybrid Attention V3 | 604,488 | Mixed CNN + LSTM + attention |
| Temporal Fusion Transformer V3 | 603,464 | Multi-component gating + attention |

Interpretation: Larger parameter counts generally imply higher training energy; runtime also influenced by sequence length, I/O, and layer type (e.g., attention quadratic cost vs linear conv cost).

## 6. CarbonCast Family Focus

CarbonCast (Base → V2 → V3) evolves CNN+LSTM fusion depth and regularization. Emissions relevance:

- CNN layers: Cheap relative to large dense/attention layers; memory locality good.
- LSTM layers: Sequential compute bottleneck; energy scales near-linearly with hidden size × sequence length.
- V3 expansions (if present) would raise energy via deeper recurrent or added channels; balancing dropout helps convergence speed (fewer epochs) which can counteract per-epoch cost.

## 7. Benchmark Aggregation & Visualization

Implemented in `benchmark.py`:
| Function | Purpose |
|----------|---------|
| `collect_emissions()` | Walks emissions directory, loads per-run totals |
| `collect_metrics_from_reports()` | Reads metrics JSON → DataFrame |
| `merge_metrics_emissions()` | Left-joins metrics with emissions for trade-offs |
| `plot_box_all_models()` | Distribution of kg CO₂e per model across runs |
| `plot_heatmaps()` | Mean kg CO₂e by model × country × season |
| `plot_tradeoff_scatter()` | Accuracy vs emissions scatter (Pareto hints) |
| `plot_top_models_bar()` | Lowest mean MAE bar ranking (indirectly energy relevant if correlated with complexity) |

### 7.1 Low-Carbon Leader Tables

`Paper/generate_tables.py` identifies “low-carbon leaders” with thresholds (e.g., <= 0.10 kg) and constructs small LaTeX tables for publication. Pareto front logic: filters by both MAE and emissions simultaneously, selecting non-dominated points (improving one metric without worsening the other).

## 8. Hypothetical Estimation Example (Illustrative Only)

Suppose a Transformer V2 run produced a CSV with final cumulative values:

```
emissions = 0.085 kg
energy_consumed = 0.42 kWh
duration = 1800 s (0.5 h)
```

Back-of-envelope validation: If regional grid intensity ≈ 200 g CO₂e/kWh (0.2 kg/kWh), then expected emissions ≈ 0.42 \* 0.2 = 0.084 kg (close to measured 0.085 kg). Minor delta can arise from embodied assumptions or small overhead in measurement windows.

Scaling to 10 runs: 10 × 0.085 kg = 0.85 kg CO₂e.
If optimization reduces per-run energy by 25%: New per-run ≈ 0.06375 kg → Savings = 0.2125 kg over 10 runs.

## 9. Estimation Methodology (When CSV Missing)

1. Approximate FLOPs or parameter-memory traffic per forward pass.
2. Multiply by (batches × epochs) to estimate total operations.
3. Convert operations to energy using hardware efficiency (e.g., modern GPU ~0.15 nJ/FLOP; CPU ~1–5 nJ/FLOP depending on vectorization).
4. Add static overhead (I/O, data loading) ~5–15% for small models.
5. Apply regional carbon intensity (kg CO₂e per kWh). Example formula:
   $$ \text{Emissions (kg)} = \frac{\text{Ops} \times \text{Energy per Op (J)}}{3.6\times10^6}\times \text{Carbon Intensity (kg/kWh)} $$
6. Calibrate with at least one empirical CodeCarbon run to adjust constants.

## 10. Optimization Opportunities

| Category          | Action                                       | Impact                                      | Effort      |
| ----------------- | -------------------------------------------- | ------------------------------------------- | ----------- |
| Data              | Cache preprocessed sequences                 | Reduce redundant scaling/slicing CPU energy | Low         |
| Training          | Early stopping (already present)             | Shortens wasted epochs                      | Low         |
| Architecture      | Replace over-wide embeddings (e.g., 256→160) | Lower per-batch MACs                        | Medium      |
| Regularization    | Increase regularization to converge earlier  | Fewer epochs                                | Low         |
| Mixed Precision   | Enable `tf.keras.mixed_precision` on GPU     | 1.3–1.8× throughput                         | Medium      |
| Profiling         | Use TensorFlow profiler to find bottlenecks  | Targets hotspot layers                      | Medium      |
| Scheduling        | Off-peak training (lower grid intensity)     | Lower actual carbon factor                  | Operational |
| Batching          | Optimize batch size for hardware sweet spot  | Improves utilization                        | Low         |
| Pruning           | Structured prune after convergence           | Smaller inference energy                    | Medium      |
| Knowledge Distill | Distill large → small model (e.g., V3→V2)    | Retain accuracy at lower cost               | Medium      |

## 11. Governance & Reproducibility

- Versioning: Keep `codecarbon.config` under version control (already present).
- Metadata: Add run metadata JSON summarizing hardware, commit SHA, dependency hash.
- Threshold Alerts: Implement guardrails (fail CI if emissions per epoch exceed baseline + X%).
- Reporting Cadence: Regenerate aggregated CSV & plots nightly or per significant model change.

## 12. Suggested Enhancements (Future Work)

| Enhancement                 | Description                                         | Benefit                                  |
| --------------------------- | --------------------------------------------------- | ---------------------------------------- |
| Per-Epoch Emissions Overlay | Merge training loss curve with cumulative emissions | Visual trade-off (marginal gain vs cost) |
| Normalized Metrics          | MAE per kg CO₂e, R² per kWh                         | Fair cross-model comparison              |
| Auto Pareto Selection       | Script outputs recommended “efficient frontier”     | Decision support                         |
| Carbon Budgeting            | Allow user to set run budget (kg)                   | Enforce sustainability goals             |
| Cluster Intensity API       | Pull live carbon intensity (e.g., ElectricityMap)   | Real-time adjustment                     |
| Multi-Device Split          | Attribute CPU vs GPU vs memory                      | Granular optimization                    |
| Emissions in Reports        | Embed final kg CO₂e in model summary markdown       | Unified documentation                    |

## 13. Risks & Mitigations

| Risk                  | Description                                 | Mitigation                                           |
| --------------------- | ------------------------------------------- | ---------------------------------------------------- |
| Missing CSVs          | Tracker not invoked or early crash          | Wrap training in resilient context; pre-flight check |
| Non-monotonic columns | Partial writes / restarts                   | Monotonicity check (already implemented)             |
| Parameter Drift       | Large embedding inflation unnoticed         | Add CI guard comparing param counts                  |
| Regional Mismatch     | Wrong carbon intensity (default DEU)        | Set correct country in config or env                 |
| Overhead Distortion   | Very short runs dominated by initialization | Aggregate multiple micro-runs                        |

## 14. Actionable Checklist

- [ ] Confirm all model scripts executed via `emissions_tracker`.
- [ ] Add hardware metadata (GPU model / CPU TDP) to CSV post-processing.
- [ ] Introduce MAE per kg CO₂e metric in benchmark outputs.
- [ ] Enable mixed precision where numerically stable.
- [ ] Set emissions regression test (max allowed increase vs baseline).
- [ ] Add one calibration run per major architecture family monthly.
- [ ] Publish “efficient frontier” plot (MAE vs kg) in README or paper.

## 15. Quick FAQ

**Q:** Why do some models with fewer params emit more?  
**A:** Longer wall-clock time (slower layers), inefficient batching, or more epochs can outweigh raw parameter count.  
**Q:** Is parameter count a reliable proxy for energy?  
**A:** Only partially; layer type and memory access patterns matter.  
**Q:** Can we retroactively estimate emissions for past runs?  
**A:** Approximate using operation counts + calibrated intensity.  
**Q:** How to reduce emissions fast?  
**A:** Early stopping + mixed precision + embedding/FF dimension tuning.

## 16. References

- CodeCarbon Documentation: https://mlco2.github.io/codecarbon/
- Energy/Carbon Optimization Patterns in ML: (industry whitepapers; add citations if publishing)

---

Factual sections derived from repository code; hypothetical example clearly labeled (Section 8). Regenerate after major architectural or infrastructure changes.
