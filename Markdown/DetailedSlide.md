# 72‑Hour Renewable Forecast — Plan Greener AI Training (Detailed Slides)

Time: ~10 minutes • Audience: mixed technical/non‑technical • Style: simple, story‑driven

---

## Slide 1 — The Big Idea (hook)

- We plan AI like a road trip: leave when the road is clear and the weather is good.
- Some hours are “green” (wind/solar high). Others are “grey” (fossil heavy).
- A 72‑hour forecast helps us pick greener hours to train heavy models.
- Visual: calendar with green/yellow/red hour blocks.

Speaker notes

- Use the road‑trip analogy. Emphasize picking the right time, not just the right model.
- Promise: same accuracy, lower cost and emissions by choosing better hours.

---

## Slide 2 — Why now? (context)

- AI workloads are growing; energy and emissions matter.
- Renewable supply is highly variable (e.g., Germany, hour‑by‑hour swings).
- Seeing 72 hours ahead = better timing, greener choices.
- Visual: line chart with “clean” and “dirty” hour highlights.

Speaker notes

- Frame the pain: volatile prices, climate goals, budget pressure.

---

## Slide 3 — Project goal (plain language)

- Predict next 72 hours of: renewable energy % and carbon intensity.
- Output: properly evaluated model with low hardware needs.
- Use prediction to schedule compute when the grid is greenest.
- Visual: 72‑hour forecast strip with highs/lows.

Speaker notes

- Keep words simple: “we forecast green hours so we can train then.”

---

## Slide 4 — What we built (workflow overview)

- Script‑only project; easy to run (no notebooks in production).
- Seasonal runners: `main_summer.py`, `main_winter.py`.
- Choose models and countries from the command line; repeat and time‑limit runs.
- Visual: Data → Models → Results & Emissions (simple flow).

Speaker notes

- Mention standardized outputs: images, metrics, emissions, summaries.

---

## Slide 5 — The data (trust first)

- 5 years of hourly data for EU countries.
- Season splits: summer and winter, per country.
- Time‑zone safe windows; robust API with fallback.
- Visual: EU map with country codes.

Speaker notes

- Explain “per‑country, per‑season” gives regional insights (green windows differ by place and season).

---

## Slide 6 — The models (simple, diverse)

- Classics: LSTM/CycleLSTM, DLinear, N‑BEATS.
- Transformers family: Transformer, TFT, Autoformer, Informer, PatchTST.
- Hybrids: CNN‑LSTM, Hybrid CNN–CycleLSTM + Attention, EnsembleCI, Mamba.
- Visual: grid of model cards (v1/v2/v3).

Speaker notes

- One line per family: “LSTMs remember sequences”, “Transformers focus on important time parts”.

---

## Slide 7 — How it runs (simple controls)

- Filter by model name; select countries; pick quick mode; set timeouts.
- Runners handle seasons and set country codes automatically.
- Visual: CLI snippet with arrows to options.

Speaker notes

- “Quick mode” = small epochs for fast demos. Timeouts keep batches moving.

---

## Slide 8 — What we save (so you can trust it)

- Plots: forecasts per model/country/season.
- Metrics: MAE, MSE, RMSE, R² (long‑horizon aware).
- Emissions: per‑run CSV via CodeCarbon.
- Visual: folder tree for Results/images, metrics, emissions.

Speaker notes

- Stress traceability: every run leaves plots, numbers, and emissions records.

---

## Slide 9 — Measuring impact (CodeCarbon)

- Tracks energy use and emissions per run.
- Scoped by model/country/season for fair comparison.
- On Windows: power estimation works; sensors (Intel Power Gadget) improve accuracy.
- Visual: small CO₂ badge per run.

Speaker notes

- “We measure so we can improve; we compare apples to apples.”

---

## Slide 10 — Sample outcome (Germany)

- Summer and winter pipelines run end‑to‑end.
- Artifacts saved: plots, metrics.json, emissions.csv, summaries.
- Use these to spot greener windows and plan training.
- Visual: side‑by‑side summer vs winter snapshots.

Speaker notes

- Share a quick real metric (e.g., MAE/ RMSE) and an example green window.

---

## Slide 11 — What you get (benefits)

- A “weather report” for clean energy hours.
- Actionable scheduling for heavy training jobs.
- Lower cost, lower emissions, same model quality.
- Visual: calendar with training blocks on green hours.

Speaker notes

- Tie benefits to teams: MLOps, researchers, finance, sustainability.

---

## Slide 12 — Limits (honest and practical)

- Windows power estimation may be noisy; sensors recommended.
- Quick mode/timeout trades speed for depth.
- Minimal HPO; baseline focus is workflow + viability.
- Visual: caution icons with short notes.

Speaker notes

- “We’re transparent. These are solvable with small next steps.”

---

## Slide 13 — Roadmap (next steps)

- Exact‑match model selection; early stopping; mixed precision.
- Carbon‑intensity‑aware scheduling signals.
- Data caching & versioning; energy benchmark suite; accelerator profiling.
- Visual: 3‑step path with milestones.

Speaker notes

- Invite contributions; small changes, real gains.

---

## Slide 14 — Call to action

- Use the forecast to plan one training window this week.
- Start with one country; extend to more as needed.
- Measure every run with CodeCarbon; share the results.
- Visual: “Start → Measure → Share” loop.

Speaker notes

- Make the ask concrete and low‑friction.

---

## Slide 15 — Backup: how to run (for Q&A)

- Summer: `python main_summer.py --countries DE --quick`
- Winter: `python main_winter.py --countries DE --quick`
- Add: `--filter Transformer` and `--timeout 300` as needed.
- Visual: command with labels (country, filter, quick).

Speaker notes

- Keep this as a reference slide for questions.

---

## Slide 16 — Backup: where files go

- Results/images/<Model>/<Country>/<Season>/
- Results/metrics/<Model>/<Country>/<Season>/metrics.json
- Results/reports/<model|season summaries>.json
- Results/emissions/<Model>/<Country>/<Season>/<run>\_emissions.csv

Speaker notes

- Reinforce transparency and repeatability.

---

## Slide 17 — Backup: model families (41 variants)

- 14 families × versions (mostly v1/v2/v3; one family has v1/v2).
- Coverage: LSTM/CycleLSTM, DLinear, N‑BEATS, Transformers (Transformer, TFT, Autoformer, Informer, PatchTST), CNN‑LSTM, Hybrid, EnsembleCI, Mamba, Robust Improved Hybrid.
- See `ModelDescription.md` for a short summary per family.

Speaker notes

- Variety enables robust comparisons across regions and seasons.

---

## Slide 18 — Backup: glossary (plain terms)

- Renewable %: share of electricity coming from renewables.
- Carbon intensity: grams of CO₂ per kWh.
- Green hours: higher renewable %, lower carbon intensity.
- Quick mode: fewer epochs; faster results; lower compute.

Speaker notes

- Use if the audience is new to energy terms.

---

## Slide 19 — Backup: risks & mitigations

- Measurement noise → add sensors, cross‑check on Linux.
- Data gaps → API fallback, caching.
- Heavy models → early stopping, mixed precision, timeouts.
- Adoption → keep CLI simple, share summaries.

Speaker notes

- Show a pragmatic path to production readiness.

---

## Slide 20 — Thank you

- One line: “Let’s train when the wind is on our side.”
- Contact / repo link (optional).

Speaker notes

- Invite a quick first pilot: one model, one country, one week.
