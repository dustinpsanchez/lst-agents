import asyncio
import types

from lst_agents import hooks
from lst_agents.store import Audit, Manifest, Store
from lst_agents.tools import full


def make_ctx(session, stage='plan', **extra):
    run = 'test-run'
    store = Store(session.bucket, session.fences, run)
    data = {'run_id': run, 'stage': stage, 'tasks': {}, 'requests': [], 'approvals': {}, 'excluded': {},
            'pending_approval': None, 'selection': {'selected': None, 'eligible': [], 'passes': {}, 'pilot': []},
            'agent_runs': {}, 'agent_cost_usd': 0.0, 'notes': [], 'history': [], 'outputs': {}, **extra}
    m = Manifest(store, run, data)
    m.save()
    return types.SimpleNamespace(m=m, audit=Audit(store, run, session.scratch_dir), run_id=run, s=session)


def test_decide_rules(session):
    ctx = make_ctx(session)
    st = {'delegations': 0}
    assert hooks.decide(ctx, 'Bash', 'orchestrator', {}, st)
    assert hooks.decide(ctx, 'mcp__other__x', 'orchestrator', {}, st)
    assert hooks.decide(ctx, 'Agent', 'orchestrator', {'subagent_type': 'general-purpose'}, st)
    assert hooks.decide(ctx, 'Agent', 'data-scout', {'subagent_type': 'qa-validator'}, st)
    assert hooks.decide(ctx, 'Agent', 'orchestrator', {'subagent_type': 'data-scout', 'run_in_background': True}, st)
    assert hooks.decide(ctx, 'Agent', 'orchestrator', {'subagent_type': 'data-scout'}, st) is None
    assert hooks.decide(ctx, full('start_maps'), 'data-scout', {'scope': 'rest'}, st)        # not its tool
    assert hooks.decide(ctx, full('start_stacks'), 'feature-builder', {'scope': 'pilot'}, st)  # wrong stage
    assert hooks.decide(ctx, full('select_passes'), 'data-scout',
                        {'drop': [{'date': '2023/07/01', 'reason': 'x'}], 'note': ''}, st)    # bad date
    assert hooks.decide(ctx, full('select_passes'), 'data-scout', {'drop': [], 'note': 'ok'}, st) is None
    for _ in range(5):
        hooks.decide(ctx, 'Agent', 'orchestrator', {'subagent_type': 'data-scout'}, st)
    assert 'at most' in hooks.decide(ctx, 'Agent', 'orchestrator', {'subagent_type': 'data-scout'}, st)


def test_pre_hook_denies_and_logs(session):
    ctx = make_ctx(session)
    hk, can_use, state = hooks.make_guards(ctx)
    pre = hk['PreToolUse'][0].hooks[0]
    out = asyncio.run(pre({'tool_name': 'Read', 'tool_input': {'file_path': '/etc/passwd'}}, None, None))
    assert out['hookSpecificOutput']['permissionDecision'] == 'deny'
    ok = asyncio.run(pre({'tool_name': full('run_status'), 'agent_type': 'feature-builder', 'tool_input': {}},
                         None, None))
    assert ok == {}
    events = [e['event'] for e in ctx.audit.tail()]
    assert events.count('denied') == 1
    res = asyncio.run(can_use('WebFetch', {'url': 'x'}, None))
    assert res.behavior == 'deny'
    res = asyncio.run(can_use(full('run_status'), {}, None))
    assert res.behavior == 'allow'


def test_session_stops_after_many_refusals(session):
    ctx = make_ctx(session)
    hk, _, state = hooks.make_guards(ctx)
    pre = hk['PreToolUse'][0].hooks[0]
    outs = [asyncio.run(pre({'tool_name': 'Bash', 'tool_input': {}}, None, None)) for _ in range(hooks.MAX_DENIALS)]
    assert outs[-1].get('continue_') is False and 'continue_' not in outs[0]
