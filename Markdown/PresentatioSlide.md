# 72-Hour Renewable Forecast — Make AI Training Greener

Slide 1 — The Big Idea

- We plan AI like a road trip: pick the best time to go
- Some hours are clean (wind/solar high), others are not
- A simple forecast tells us when to train heavy models
- Visual idea: calendar with green/yellow/red hours

Slide 2 — Why This Matters (Now)

- AI workloads growing fast; energy cost and emissions matter
- Renewable energy changes hour to hour (esp. Germany)
- If we can see 72 hours ahead, we can choose greener hours
- Visual idea: line chart with “clean” and “dirty” hour highlights

Slide 3 — Project Goal

- Forecast next 72 hours: renewable energy % and carbon intensity
- Output: a properly evaluated model with low hardware needs
- Use results to schedule compute when the grid is green
- Visual idea: 72-hour forecast strip showing highs and lows

Slide 4 — What We Built (Workflow)

- Script-only project (no notebooks in production)
- Two runners: `main_summer.py`, `main_winter.py`
- Run by country list; dynamic COUNTRY_CODE; easy CLI
- Visual idea: simple flow diagram: Data → Models → Results/Emissions

Slide 5 — The Data (Clear & Trustworthy)

- 5 years of hourly data for EU countries
- Season splits: summer and winter, per country
- Safe time windows; API with robust fallback
- Visual idea: map of EU with country codes

Slide 6 — The Models (Diverse, Tested)

- Classics: LSTM/CycleLSTM, DLinear, N-BEATS
- Transformers family: Transformer, TFT, Autoformer, Informer, PatchTST
- Hybrids: CNN-LSTM, Hybrid CNN–CycleLSTM + Attention, EnsembleCI, Mamba
- Visual idea: grid of model “cards” (families + v1/v2/v3)

Slide 7 — How It Runs (Simple Controls)

- Filter by model name; choose countries; repeat runs
- Quick mode (small epochs) for fast experiments
- Optional timeouts keep batches moving
- Visual idea: CLI snippet with arrows to settings

Slide 8 — What We Save (So You Can Trust It)

- Plots: forecast visuals per model/country/season
- Metrics: MAE, MSE, RMSE, R² (long-horizon aware)
- Emissions: CodeCarbon CSV per run
- Visual idea: folder tree of Results/images, metrics, emissions

Slide 9 — CodeCarbon (Measure to Improve)

- Tracks energy use and carbon emissions per run
- Country/season/model scoping for fair comparison
- Windows note: power estimation works; sensors improve accuracy
- Visual idea: small CO₂ badge per run

Slide 10 — Sample Outcome (Germany)

- Summer and winter pipelines run end-to-end
- Artifacts saved automatically: plots, metrics.json, emissions.csv
- Summaries for quick reporting
- Visual idea: side-by-side summer vs winter plot snapshots

Slide 11 — What You Get

- A “weather report” for clean energy hours
- Actionable: schedule heavy training when greener
- Lower cost, lower emissions, same model quality
- Visual idea: calendar with training blocks placed on green hours

Slide 12 — Limits (Honest & Practical)

- Windows power estimation; sensors recommended
- Quick mode/timeout trades speed for depth
- Minimal HPO in baseline; results focus on workflow + viability
- Visual idea: caution icons with short notes

Slide 13 — Roadmap (What’s Next)

- Exact-match model selection; better auto-stopping
- Mixed precision; accelerator profiling
- Carbon-intensity-aware scheduling signals
- Data caching and versioning; energy benchmark suite
- Visual idea: 3-step path with milestones

Slide 14 — Call to Action

- Use the forecast to plan training windows this week
- Start with one country; extend to your fleet
- Measure with CodeCarbon; share the results
- Visual idea: “Start → Measure → Share” loop

Slide 15 — Backup (How to Run)

- Summer: `python main_summer.py --countries DE --quick`
- Winter: `python main_winter.py --countries DE --quick`
- Add: `--filter Transformer` or `--timeout 300`
- Visual idea: command + short labels (country, filter, quick)

Notes for presenter

- Keep language simple; focus on the “why” before the “how”
- Use visuals: calendar (timing), charts (variation), folder tree (trust)
- Emphasize benefits: greener, cheaper, smarter
- Close with a small ask: try one scheduled run and report emissions
