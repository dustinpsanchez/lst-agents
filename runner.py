"""One agent session per stage, with every fence switched on."""
from __future__ import annotations

import asyncio
import os

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, TextBlock,
                              ToolUseBlock)

from . import agents, pipeline, tools
from .hooks import AGENT_TOOL_NAMES, make_guards
from .store import clip, now_utc

# Built-in tools the agents must never have. tools=['Agent'] already leaves them out; this list is the second lock.
BLOCKED_BUILTINS = ['Bash', 'BashOutput', 'KillShell', 'Read', 'Write', 'Edit', 'MultiEdit', 'NotebookEdit',
                    'Glob', 'Grep', 'WebSearch', 'WebFetch', 'TodoWrite', 'Skill', 'SlashCommand', 'ToolSearch',
                    'ListMcpResourcesTool', 'ReadMcpResourceTool', 'Monitor', 'CronCreate', 'CronDelete',
                    'CronList', 'TaskCreate', 'TaskUpdate', 'TaskList', 'TaskGet', 'TaskStop', 'TaskOutput',
                    'SendMessage', 'AskUserQuestion', 'EnterPlanMode', 'ExitPlanMode', 'EnterWorktree',
                    'ExitWorktree', 'LSP']

CLI_ENV = {
    'CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH': '1',     # specialists can't start agents of their own
    'CLAUDE_CODE_DISABLE_BACKGROUND_TASKS': '1',     # specialists finish before the orchestrator continues
    'CLAUDE_CODE_DISABLE_MCP_TASK_BACKGROUND': '1',  # tool calls finish before the agent continues
    'CLAUDE_CODE_DISABLE_EXPLORE_PLAN_AGENTS': '1',  # no built-in Explore/Plan agents
    'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', # no telemetry or auto-update calls
    'DISABLE_AUTOUPDATER': '1',
    'MCP_TOOL_TIMEOUT': '600000',
}


def options_for(ctx, stage: str, budget_usd: float, state: dict) -> ClaudeAgentOptions:
    server, _ = tools.build_server(ctx)
    hooks, can_use_tool, _ = make_guards(ctx, state)
    scratch = os.path.join(ctx.s.scratch_dir, ctx.run_id)
    os.makedirs(scratch, exist_ok=True)
    return ClaudeAgentOptions(
        tools=['Agent'],                       # the only built-in tool: delegation
        allowed_tools=[],                      # nothing is pre-approved; can_use_tool decides
        disallowed_tools=BLOCKED_BUILTINS,
        mcp_servers={tools.SERVER: server},
        strict_mcp_config=True,                # no MCP servers from any config file
        setting_sources=[],                    # no settings, hooks or agent files from disk
        agents=agents.definitions(ctx.job),
        hooks=hooks,
        can_use_tool=can_use_tool,
        system_prompt=agents.ORCHESTRATOR_PROMPT,
        model=ctx.job['models']['orchestrator'],
        max_turns=24,
        max_budget_usd=budget_usd,
        cwd=scratch,                           # the runtime's own disk, never Drive
        env={**CLI_ENV, 'ANTHROPIC_API_KEY': ctx.s.anthropic_key},
    )


async def run_stage_session(ctx, stage: str, say=print) -> dict:
    """Runs the orchestrator for one stage. Returns the session record saved in the manifest."""
    m, gates = ctx.m, ctx.job['gates']
    remaining = gates['agent_budget_usd'] - m['agent_cost_usd']
    if remaining < 0.10:
        raise pipeline.StopRun(f"The Claude budget for this run (${gates['agent_budget_usd']}) is used up "
                               f"(${m['agent_cost_usd']:.2f} spent). To continue, raise it in a new cell with "
                               f"lst.set_budget(S, RUN_ID, <dollars>) and run the advance cell again.")
    budget = round(min(gates['session_budget_usd'], remaining), 2)
    state: dict = {}
    options = options_for(ctx, stage, budget, state)
    brief = agents.stage_brief(ctx, stage)
    ctx.audit.log('session_start', stage=stage, budget_usd=budget)
    say(f'· Agents working on: {pipeline.STAGE_LABEL[stage]} (budget ${budget:.2f})')
    subagent_of: dict[str, str] = {}
    final_text, result = '', None
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(brief)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    who = subagent_of.get(msg.parent_tool_use_id or '', 'orchestrator')
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            if block.name in AGENT_TOOL_NAMES:
                                sub = (block.input or {}).get('subagent_type', '?')
                                subagent_of[block.id] = sub
                                say(f'    {who} → delegates to {sub}')
                            else:
                                say(f"    {who} → {block.name.split('__')[-1]}")
                        elif isinstance(block, TextBlock) and who == 'orchestrator':
                            final_text = block.text
                elif isinstance(msg, ResultMessage):
                    result = msg
    except Exception as err:
        ctx.audit.log('session_error', stage=stage, error=clip(str(err), 400))
        m['agent_runs'][stage] = m['agent_runs'].get(stage, 0) + 1
        ctx.save()
        say(f'  The agent session stopped with an error: {clip(str(err), 300)}')
        return {'stage': stage, 'error': str(err)}
    cost = float(result.total_cost_usd or 0) if result else 0.0
    rec = {'stage': stage, 'at': now_utc(), 'cost_usd': round(cost, 4),
           'turns': result.num_turns if result else None, 'outcome': result.subtype if result else 'no result',
           'tool_calls': state.get('calls', 0), 'refusals': state.get('denials', 0),
           'session_id': result.session_id if result else None, 'summary': clip(final_text, 1500)}
    m['agent_cost_usd'] = round(m['agent_cost_usd'] + cost, 4)
    m['agent_runs'][stage] = m['agent_runs'].get(stage, 0) + 1
    m['sessions'].append(rec)
    ctx.save()
    ctx.audit.log('session_end', **{k: v for k, v in rec.items() if k != 'summary'})
    if final_text:
        say('\n' + final_text.strip() + '\n')
    say(f"  (session cost ${cost:.3f}; run total ${m['agent_cost_usd']:.2f} of ${gates['agent_budget_usd']:.2f})")
    return rec
