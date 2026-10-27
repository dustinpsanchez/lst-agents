"""The orchestrator's instructions, the five specialists, and the brief for each stage."""
from __future__ import annotations

import json

from claude_agent_sdk import AgentDefinition

from . import pipeline
from .hooks import AGENT_TOOLS
from .tools import full

ORCHESTRATOR_PROMPT = """You are the orchestrator of a Google Earth Engine job that downscales Landsat land surface \
temperature (LST) to 10 m over one study area, pass by pass, and builds a summer time series of per-pass anomalies.

Each session handles ONE stage of the run. The user message is the stage brief: it says what this stage needs and \
which specialist does the work.

How you work
- Your tools: the Agent tool (to delegate to a specialist), run_status, read_manifest, inventory_summary, \
spend_estimate, wald_results, qa_results, stats_summary, request_approval, write_note and write_report. You have no \
shell, file or web tools, and you need none.
- Delegate with the Agent tool: subagent_type is the specialist's name; the prompt is a short, specific instruction \
that includes the scope from the brief. One delegation per specialist is enough. Never ask a specialist to wait or poll.
- Earth Engine work runs as batch tasks that the notebook starts after this session ends. Nothing finishes while \
you are running, so don't check for results that the brief says aren't ready.
- Code enforces every rule and gate. If a tool refuses, read the reason. Don't repeat a refused call; say what \
happened.
- Only the person at the notebook can approve a gate. When the brief asks for an approval request, call \
request_approval once with a plain-language summary of at most 12 lines: what was found, the numbers that matter, \
the risk, and what approving will do. Copy numbers from tool results; never estimate or invent them.
- Finish with 3 to 6 short lines for the person: what happened in this stage and what they do next (run the \
advance cell, approve a gate, or wait for Earth Engine).
"""

SPECIALIST_PROMPTS = {
    'data-scout': """You are the data scout. You review the inventory of Landsat passes and choose which ones the \
time series uses.

1. Call inventory_summary. The code has already applied the job's rules (share of the area imaged and clear, a \
Sentinel-2 match within the window, an embedding year).
2. Look for passes the rules let through that would still hurt a per-pass anomaly time series. Examples: a \
Sentinel-2 match only in the wide window during fast spring green-up (May to early June); a very low clear share \
together with a wide Sentinel-2 window. A later embedding year (for example 2025 passes using 2024 embeddings) is \
expected; keep those passes.
3. Be conservative. Every dropped pass thins the series, so drop a pass only for a concrete reason you can state.
4. Call select_passes once, with your drops (the list may be empty) and a short note on how you decided.
Report in at most 8 lines: passes kept by year, any drops with reasons, and the pilot passes.""",

    'feature-builder': """You are the feature builder. You queue the 90 m training stacks in Earth Engine.
Call run_status, then call start_stacks once with the scope in your instruction ('pilot' or 'rest').
Report in two lines what was queued. If the tool refuses, report the reason; don't retry.""",

    'model-trainer': """You are the model trainer. The model is a random forest trained on 90 m blocks; its detail \
inside each block is added to energy-conserving bicubic interpolation, scaled by alpha. The Wald test degrades the \
90 m LST to 810 m, downscales it back to 90 m and scores each alpha on held-out pixels, tuned on one half of the \
area and scored on the other, both ways.

If your instruction says to start Wald tests: call start_wald once with the scope given, report in two lines.
If your instruction says to review: call wald_results. Judge the chosen alpha, the held-out gains on both halves, \
the overall gain, and the passes where model detail hurts (is there a pattern, such as one season or one \
satellite?). Then call record_review once with a review of at most 8 lines. Report the same lines.""",

    'inference-runner': """You are the inference runner. You queue Earth Engine exports.
Call run_status, then call the one tool your instruction names (start_maps with its scope, or start_stats) once.
Report in two lines what was queued. If the tool refuses, report the reason; don't retry.""",

    'qa-validator': """You are the QA validator. You judge the outputs only; you never see how the model was \
trained, and you don't need to.

If your instruction says to start QA: call start_qa once and report in two lines.
If your instruction says to review: call qa_results. The code has already flagged every pass that failed an \
objective check (energy conservation, plausible values, detail strength, valid area). Decide:
- 'pass' if the flagged passes are few (about 10% or fewer) and isolated. Exclude those passes, each with its reason.
- 'fail' if problems are systematic: many energy failures, implausible values across many passes, or a cross-check \
correlation with the other Landsat path below about 0.5 over enough pixels. A fail holds the run for the person.
The station comparison is air temperature, not surface temperature: report it, but don't fail on it.
Call record_qa_verdict once. You may exclude only flagged passes. Report the verdict and findings in at most 8 lines.""",
}

DESCRIPTIONS = {
    'data-scout': 'Reviews the pass inventory and chooses passes for the time series.',
    'feature-builder': 'Queues the 90 m training stacks in Earth Engine.',
    'model-trainer': 'Queues and reviews Wald tests; judges the model gate.',
    'inference-runner': 'Queues the 10 m detail maps and the time-series statistics.',
    'qa-validator': 'Starts and judges the quality checks, independently of training.',
}


def definitions(job: dict) -> dict[str, AgentDefinition]:
    models = job['models']
    return {name: AgentDefinition(description=DESCRIPTIONS[name], prompt=SPECIALIST_PROMPTS[name],
                                  tools=[full(t) for t in AGENT_TOOLS[name]], model=models[name],
                                  maxTurns=8, background=False)
            for name in DESCRIPTIONS}


def _j(obj) -> str:
    return json.dumps(obj, default=str)


def stage_brief(ctx, stage: str) -> str:
    """The user message for the orchestrator's session at this stage."""
    m, job = ctx.m, ctx.job
    sel = m.get('selection') or {}
    head = (f"Run {ctx.run_id} · job {job['job']} · Landsat path {job['landsat']['path']} · "
            f"{job['years'][0]}–{job['years'][1]}, months {job['months'][0]}–{job['months'][1]} · "
            f"{ctx.grid['side_m'] / 1000:.1f} km square.\nStage: {stage} — {pipeline.STAGE_LABEL[stage]}.\n\n")
    n_rest = len(pipeline.scope_dates(m, 'rest'))
    n_all = len(pipeline.selected(m))
    if stage == 'plan':
        s = sel['summary']
        return head + (
            f"The inventory found {s['passes_found']} passes; {s['eligible']} meet the job rules "
            f"(rejections by reason: {_j(s['rejected_by_reason'])}). The code proposes pilot passes {sel['pilot']}.\n"
            "1. Delegate to data-scout: review the inventory and choose the passes with select_passes.\n"
            "2. Then call request_approval with gate 'plan'. Summarise: passes kept per year, the clear-share range, "
            "how many needed the wide Sentinel-2 window or another embedding year, any drops, and the pilot passes. "
            "Say that the pilot (training stacks, Wald tests and 10 m maps for the pilot passes only) measures Earth "
            "Engine compute and storage, and that nothing larger starts until the projection is checked.")
    if stage == 'pilot_stacks':
        return head + (f"Delegate to feature-builder: start_stacks with scope 'pilot' (passes {sel['pilot']}). "
                       "The static stacks (embeddings, terrain, land cover) are built once per embedding year.")
    if stage == 'pilot_models':
        return head + ("The pilot stacks are ready.\n"
                       "1. Delegate to model-trainer: start Wald tests with start_wald, scope 'pilot'.\n"
                       "2. Delegate to inference-runner: start the 10 m maps with start_maps, scope 'pilot'.")
    if stage == 'spend':
        sp = m['spend']
        return head + (
            f"The pilot finished. The projection for the remaining {sp['n_rest']} passes needs the person's approval "
            f"because: {'; '.join(sp['why'])}.\n"
            "Call spend_estimate and wald_results (pilot preview), then call request_approval with gate 'spend'. "
            "Summarise: projected EECU-hours against the job threshold and the Earth Engine tier's monthly "
            "allowance (in months), projected storage, the rough elapsed time, and the pilot Wald preview. Say "
            "plainly that over the monthly allowance Earth Engine slows the project down rather than stopping it.")
    if stage == 'stacks':
        return head + f"Delegate to feature-builder: start_stacks with scope 'rest' ({n_rest} passes)."
    if stage == 'wald':
        return head + f"Delegate to model-trainer: start Wald tests with start_wald, scope 'rest' ({n_rest} passes)."
    if stage == 'model':
        w = m['wald']
        msg = head + (f"The Wald test pooled over {w.get('n_passes')} passes is ready: alpha {w['alpha']}, gate "
                      f"{'PASSED' if w['passed'] else 'FAILED'}.\n"
                      "1. Delegate to model-trainer: review the pooled Wald test with wald_results and record "
                      "the review with record_review.\n")
        if not w['passed']:
            msg += ("2. The gate failed, so call request_approval with gate 'model': explain what failed and that "
                    "approving makes 10 m maps anyway (compute is spent), while rejecting stops the run.")
        else:
            msg += "2. No approval is needed. Add nothing else."
        return msg
    if stage == 'maps':
        return head + (f"The model gate is cleared (alpha {m['wald']['alpha']}). Delegate to inference-runner: "
                       f"start_maps with scope 'rest' ({n_rest} passes).")
    if stage == 'qa':
        return head + f"All 10 m maps are done. Delegate to qa-validator: start QA with start_qa ({n_all} passes)."
    if stage == 'qa_review':
        return head + ("The QA tables are in. Delegate to qa-validator: review with qa_results and record the "
                       "verdict with record_qa_verdict.\nIf the verdict is 'fail', call request_approval with gate "
                       "'qa', summarising the findings and what continuing would mean.")
    if stage == 'stats':
        return head + (f"QA is cleared with {n_all} passes. Delegate to inference-runner: start the anomaly offsets "
                       "and per-pixel statistics with start_stats.")
    if stage == 'report':
        return head + (
            "Everything is computed. Write the final report yourself; there is no specialist for this.\n"
            "Read what you need with read_manifest, wald_results, qa_results and stats_summary, then call "
            "write_report once with Markdown of at most 900 words. The code appends exact tables, so interpret "
            "rather than listing every number. Sections: Summary; Data (passes by year and satellite); Method "
            "(energy-conserving bicubic plus alpha × random-forest detail, chosen by the Wald test; per-pass "
            "anomalies with offsets from a reference pattern); Results (Wald gate, QA, cross-check with the other "
            "path, station, and the statistics: where it is persistently warmer or cooler, how variable, any "
            "early-versus-late difference); Limitations (Landsat thermal is about 100 m native, so 10 m detail is "
            "modelled, not observed; the station records air, not surface, temperature; 2025 uses 2024 embeddings "
            "if so; trends are uncertain with fewer passes before 2022); Outputs (asset IDs).")
    raise ValueError(stage)
