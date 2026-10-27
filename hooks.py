"""The guards around every agent session.

PreToolUse runs before every tool call, including calls made inside subagents.
It refuses anything that isn't one of this job's tools, any tool the calling
agent isn't allowed, any subagent other than the five specialists, any date that
isn't a plain date, and anything the stage machine says can't happen yet.
Every refusal and every completed call goes to runs/RUN_ID/audit.jsonl.
"""
from __future__ import annotations

import asyncio

from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

from . import pipeline
from .fences import FenceError, Fences
from .store import clip
from .tools import full

AGENT_TOOL_NAMES = ('Agent', 'Task')       # the CLI calls it Agent; Task is its older name
AGENT_TOOLS = {
    'orchestrator': ['run_status', 'read_manifest', 'inventory_summary', 'spend_estimate', 'wald_results',
                     'qa_results', 'stats_summary', 'request_approval', 'write_note', 'write_report'],
    'data-scout': ['run_status', 'inventory_summary', 'select_passes'],
    'feature-builder': ['run_status', 'start_stacks'],
    'model-trainer': ['run_status', 'start_wald', 'wald_results', 'record_review'],
    'inference-runner': ['run_status', 'start_maps', 'start_stats'],
    'qa-validator': ['run_status', 'start_qa', 'qa_results', 'record_qa_verdict', 'stats_summary'],
}
SPECIALISTS = [a for a in AGENT_TOOLS if a != 'orchestrator']
MAX_DELEGATIONS = 4          # per session; a stage needs one or two
MAX_DENIALS = 6              # after this many refusals the session is stopped


def _deny(reason: str, stop: bool = False) -> dict:
    out = {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny',
                                  'permissionDecisionReason': reason}}
    if stop:
        out['continue_'] = False
        out['stopReason'] = 'Too many refused tool calls; the session was stopped.'
    return out


def decide(ctx, name: str, agent: str, args: dict, state: dict) -> str | None:
    """Why this call must be refused, or None to let it through."""
    if name in AGENT_TOOL_NAMES:
        if agent != 'orchestrator':
            return 'specialists cannot start other agents'
        sub = (args or {}).get('subagent_type')
        if sub not in SPECIALISTS:
            return f'subagent_type must be one of {SPECIALISTS}'
        if (args or {}).get('run_in_background'):
            return 'specialists run in the foreground only'
        if state['delegations'] >= MAX_DELEGATIONS:
            return f'at most {MAX_DELEGATIONS} delegations per session'
        state['delegations'] += 1
        return None
    prefix = full('')
    if not name.startswith(prefix):
        return 'only this job\'s tools are available (no shell, files, web or other servers)'
    short = name[len(prefix):]
    if short not in AGENT_TOOLS.get(agent, []):
        return f'{short} is not one of the {agent} tools'
    try:
        for d in (args or {}).get('drop', []) + (args or {}).get('exclude', []):
            Fences.check_date(d.get('date'))
    except (FenceError, AttributeError) as err:
        return str(err)
    return pipeline.check_tool(ctx.m, short, args or {})


def make_guards(ctx, state: dict | None = None):
    """Returns (hooks dict for ClaudeAgentOptions, can_use_tool callback, state)."""
    from claude_agent_sdk import HookMatcher
    state = state if state is not None else {}
    state.setdefault('delegations', 0)
    state.setdefault('denials', 0)
    state.setdefault('calls', 0)

    def log(event, **kw):
        return asyncio.to_thread(ctx.audit.log, event, **kw)

    async def pre(inp, tool_use_id, context):
        name = inp.get('tool_name', '')
        agent = inp.get('agent_type') or 'orchestrator'
        args = inp.get('tool_input') or {}
        reason = decide(ctx, name, agent, args, state)
        if reason:
            state['denials'] += 1
            await log('denied', tool=name, agent=agent, reason=reason, input=clip(args, 300))
            return _deny(reason, stop=state['denials'] >= MAX_DENIALS)
        return {}

    async def post(inp, tool_use_id, context):
        state['calls'] += 1
        await log('tool', tool=inp.get('tool_name'), agent=inp.get('agent_type') or 'orchestrator',
                  input=clip(inp.get('tool_input'), 300), result=clip(inp.get('tool_response'), 400))
        return {}

    async def post_fail(inp, tool_use_id, context):
        await log('tool_error', tool=inp.get('tool_name'), agent=inp.get('agent_type') or 'orchestrator',
                  input=clip(inp.get('tool_input'), 300), error=clip(inp.get('error'), 300))
        return {}

    allowed = {full(t) for ts in AGENT_TOOLS.values() for t in ts} | set(AGENT_TOOL_NAMES)

    async def can_use_tool(tool_name, input_data, context):
        if tool_name in allowed:
            return PermissionResultAllow(updated_input=input_data)
        state['denials'] += 1
        await log('denied', tool=tool_name, agent='(permission check)', reason='not an allowed tool',
                  input=clip(input_data, 300))
        return PermissionResultDeny(message='Only this job\'s tools are available.', interrupt=False)

    hooks = {'PreToolUse': [HookMatcher(hooks=[pre], timeout=60)],
             'PostToolUse': [HookMatcher(hooks=[post], timeout=60)],
             'PostToolUseFailure': [HookMatcher(hooks=[post_fail], timeout=60)]}
    return hooks, can_use_tool, state
