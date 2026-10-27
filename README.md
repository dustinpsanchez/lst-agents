# lst-agents · Phase 2: the agent layer

Claude agents run a Google Earth Engine job that downscales Landsat land surface temperature (LST) to 10 m. The job runs pass by pass over the San Antonio study area and builds a summer time series of per-pass anomalies. Everything runs from one Colab notebook, `notebooks/run_job.ipynb`.

The agents plan, start steps, review results and write the report. Tested Python does the Earth Engine work and enforces every gate. Agents can't approve anything, can't touch files, the shell or the web, and can't reach anything outside the project's bucket prefix and asset folder.

## What a run does

| Stage | Who | What happens |
|---|---|---|
| inventory | code | One batch task lists every path-27 pass: how much is clear, and how well Sentinel-2 matches |
| plan | data-scout, orchestrator | The scout reviews the passes and drops any that would hurt the series. **You approve the plan.** |
| pilot | feature-builder, model-trainer, inference-runner | Two passes run end to end: 90 m stacks, Wald tests and 10 m maps. Earth Engine compute and storage are measured on them |
| spend | code (orchestrator if needed) | The pilot is projected to the full run. **You approve if it's over the job's limits** |
| stacks, wald | feature-builder, model-trainer | 90 m stacks and Wald tests (810 m → 90 m) for every other pass |
| model | model-trainer | One alpha is chosen from all passes' Wald tests. **You approve only if the gate fails** |
| maps | inference-runner | 10 m model-detail maps for every other pass |
| qa, qa_review | qa-validator | Per-pass checks: energy conservation, plausible values, detail strength, the station, and a path-28 cross-check. **You approve only after a fail** |
| stats | inference-runner | Per-pass anomaly offsets, then per-pixel statistics at 10 m |
| report | orchestrator | A Markdown report; the code appends the exact numbers |

The method is the one that passed Phase 1: 10 m LST = energy-conserving bicubic + alpha × random-forest detail. Every pass stores only the detail (int16, hundredths of a kelvin), so one alpha, chosen from every pass's Wald test, applies to the whole series.

Outputs go under `projects/PROJECT/assets/lst/RUN_ID/`:

- `stats10`: per pixel at 10 m, int16. The bands are `n_looks`, `mean_cK`, `sd_cK`, `trend_cK_per_decade`, `n_early`, `early_cK`, `n_late` and `late_cK`. Values ending `_cK` are hundredths of a kelvin.
- `detail10/`: one model-detail image per pass.
- `stack90/` and `static90/`: the 90 m stacks. Delete these with `cleanup()` once the run is done.

The report and run record go under `gs://BUCKET/lst-agents/runs/RUN_ID/`. There you'll find `report.md`, `manifest.json`, `audit.jsonl` and `tables/`.

## Put the code in your private repo

1. Unzip `lst-agents.zip`.
2. On GitHub, open your private `lst-agents` repo and choose **Add file → Upload files**.
3. Drag in everything inside the unzipped folder: the `lst_agents`, `jobs`, `notebooks` and `tests` folders, plus `README.md`, `requirements.txt` and `.gitignore`. Commit to `main`.
4. Your read-only `GITHUB_TOKEN` in Colab Secrets must have **Contents: Read** on this repo.

## Run it

1. Open `notebooks/run_job.ipynb` in Colab (File → Upload notebook) and save it to your Drive. Drive is never mounted.
2. Runtime → Change runtime type → **CPU**.
3. In the first cell, set `GITHUB_REPO` to `your-user/lst-agents`.
4. Secrets needed, with notebook access on: `LST_SA_KEY`, `ANTHROPIC_API_KEY` and `GITHUB_TOKEN`.
5. Run the **smoke test** first (`JOB_NAME = 'smoke_test'`). It covers a 16 km box and four passes from summer 2023, and checks every stage against real Earth Engine for a few EECU-hours. Then switch to `JOB_NAME = 'san_antonio_p27'`.

Each time, use **Runtime → Run all**. The notebook stops on its own at approvals and long waits. To approve, read the request it prints, set `GATE` in the approval cell, and Run all again.

## Costs

- **Claude**: each stage runs one short agent session, about $0.01 to $0.20 on Sonnet. `gates.agent_budget_usd` caps the whole run (default $5) and `session_budget_usd` caps each session.
- **Earth Engine**: noncommercial projects get a monthly EECU-hour allowance. The Community tier gives 150 and the Contributor tier 1,000. Contributor needs only a billing account on the project, and Earth Engine doesn't charge it for noncommercial use. Over the allowance, Earth Engine slows the project down rather than stopping it. The pilot measures real usage, and the spend gate stops the run before the full batch if the projection is over `gates.eecu_hours_over`.
- **Colab**: a CPU runtime is enough. Earth Engine keeps working when the notebook disconnects. `advance()` waits at most `wait_minutes` (default 30) and the last cell disconnects the runtime.

## Safety model

- The runtime signs in only as the project's service account. It never signs in as you and never mounts Drive.
- Agents have one built-in tool, `Agent`, which delegates to the five specialists. They also have the job's own tools in `lst_agents/tools.py`. Every other tool is switched off (`tools=['Agent']` plus `disallowed_tools`). `setting_sources=[]` and `strict_mcp_config=True` keep any settings, hooks or MCP servers from loading from disk.
- A `PreToolUse` hook checks every call, including calls made by specialists. It refuses any tool that isn't on the calling agent's list, any subagent that isn't one of the five, any date that isn't plain `YYYY-MM-DD`, and anything the stage machine doesn't allow yet. A `PostToolUse` hook logs every call to `audit.jsonl`, and refusals are logged too.
- Tools never take a path or an asset ID. Every name is built in `fences.py`. Writes go only into the run's folder or `exports/`, and assets only under `projects/PROJECT/assets/lst/`. No tool deletes or moves anything; `cleanup()` is yours to run.
- Agents only *request* Earth Engine work. Plain Python starts the tasks after the session ends, and only approvals you give in the notebook move a gate.
- If an agent fails to act twice, the code completes that stage with the default rules. A run never stalls on an agent, and never skips a gate.

## Tests

```
pip install claude-agent-sdk==0.2.172 earthengine-api pyyaml pandas pytest
python -m pytest -q tests
```

The tests replace Earth Engine and the bucket with stand-ins. They walk a whole run through the stage machine, check the fences and hooks, and build every Earth Engine graph to catch Python mistakes. They can't check Earth Engine semantics; that is what the smoke test is for.
