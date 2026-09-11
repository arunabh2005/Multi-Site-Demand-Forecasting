# Project Brief: Multi-Location Demand Forecasting Tool with AI-Generated Insights

## Background / who I am
I'm a beginner-to-intermediate learner in data science/ML — comfortable with basic Python, but new to time-series forecasting, clustering, and applied ML. I want to learn the concepts *while* building this, so please explain key decisions in plain language as we go, not just write code silently. I have about 10 days to build this.

## What we're building (goal)
A tool that takes historical demand data (e.g., EV charging load, water consumption, energy usage — any time-stamped numeric demand data across multiple locations/entities) and:

1. Organizes and cleans the raw data.
2. Explores it visually to understand patterns (daily/weekly cycles, trends, irregularities).
3. Groups similar locations/entities together based on how their demand behaves over time.
4. Forecasts future demand for each location/group.
5. Compares the forecasting approach against at least one simple reference method and at least one moderately stronger reference method, to honestly measure how much better (or not) the main approach is.
6. Produces a short, plain-English written summary of the results and any notable patterns/anomalies, generated automatically rather than manually written.

## Why this shape (context, not a rulebook)
This project should mirror how demand forecasting is used in the real world — for things like utilities, EV charging networks, retail, or grid operators — where the value isn't just "predict a number" but "predict reliably across many locations, know how confident to be, and explain findings to someone non-technical." Feel free to suggest a real, freely available public dataset that fits this shape (some options: EV charging session data, municipal water/energy consumption data, retail demand data) — pick whatever is realistically achievable to source and clean quickly.

## What I don't want prescribed
- Don't lock us into one specific algorithm/library up front — propose reasonable options (with trade-offs in plain language) and let's choose together based on what's easy to implement well, appropriate for the data size, and defensible if I'm asked about it later.
- Don't assume large-scale infrastructure, GPUs, or heavy deep learning — this should run comfortably on a normal laptop, on a moderately sized dataset (thousands to low hundred-thousands of rows, not millions).
- Don't just optimize for a flashy accuracy number — prioritize a defensible, explainable pipeline over a maximized metric.

## End deliverables
- A working codebase (script or notebook) with clear structure, in a GitHub-ready state.
- Visualizations of raw patterns, groupings, and forecast vs. actual results.
- A results table comparing methods with an appropriate error metric, run on genuinely held-out (future) data, not just training data.
- A short natural-language summary/report of findings, generated using an LLM API call over the pipeline's outputs.
- A README explaining: what the project does, what data it uses, what methods were tried and why, and what the actual measured results were (real numbers from the run, not assumed/target numbers).
- (Optional, nice-to-have) A simple interactive dashboard to explore results, if time allows.

## Constraints
- Timeline: ~10 days, part-time learning + building, with breaks/exams in between — assume real usable time is less than 10 full days.
- Compute: standard laptop, no GPU dependency required.
- I want to understand *why* each major step is done, not just have it run — flag the 4-5 core concepts I should understand as we build (e.g., what a baseline is for, why we split data by time, what the error metric means) so I can explain the project confidently later.

## Once the core pipeline works: optional scope extensions
If time allows after the core pipeline (cleaning → EDA → clustering → baseline(s) → main model → LLM summary) is working end-to-end, extend it properly with some of the following — please treat these as genuine engineering additions, not padding, and explain each briefly as it's added:

- **Config-driven setup**: move hardcoded values (file paths, column names, number of clusters, etc.) into a separate config file so the pipeline can be pointed at a different dataset without editing the core code.
- **Basic tests**: a few simple checks that verify key functions (data cleaning, metric calculations) behave correctly on known sample inputs.
- **Error handling / data validation**: sensible handling of missing values, malformed dates, or bad rows, so the pipeline doesn't silently break or produce wrong results on messy real-world data.
- **Logging**: simple status messages (and optionally a log file) showing what each stage of the pipeline did and how long it took, so a run's progress and any failures are traceable.
- **A proper multi-section dashboard**: rather than a single chart, a small dashboard with a few views (raw data/trends, cluster groupings, forecast vs. actual, and the generated summary) so results can be explored interactively, not just read from a script's output.

These should only be added after the core pipeline is solid — don't let them delay having a working end-to-end version first.
