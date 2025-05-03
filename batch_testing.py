# -*- coding: utf-8 -*-
"""
Batch‑tester for CMAS / DMAS / HMAS‑1 / HMAS‑2 / ETP
on BoxNet1 & BoxNet2 (5 frameworks × 2 envs).

•  live‑flushes raw results after every trial
•  writes per‑framework summary + bar‑plots
•  BoxNet2 success = fraction‑of‑colors whose goal list is empty
•  DMAS wrapper now:
     – deduplicates actions (keeps most‑recent per agent)
     – supports optional token return from DMAS
"""

import os, csv, argparse, traceback, json, re
from datetime import datetime
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ────────────────────────────────────────────────────────────
#  Environments
# ────────────────────────────────────────────────────────────
from BoxNet1 import BoxNet1
from BoxNet2_test import BoxNet2

# ────────────────────────────────────────────────────────────
#  Frameworks
# ────────────────────────────────────────────────────────────
import CMAS, DMAS, HMAS1, HMAS2, ETP

# ────────────────────────────────────────────────────────────
#  Generic parsing / execution utilities
# ────────────────────────────────────────────────────────────
_MOVE_RE  = re.compile(r"move (\w+) box from \((\d+),\s*(\d+)\).*?(\bup|\bdown|\bleft|\bright)", re.I)
_GOAL_RE  = re.compile(r"move (\w+) box to goal", re.I)
_NOTH_RE  = re.compile(r"do\s+nothing", re.I)

def _to_lines(obj) -> List[str]:
    if obj is None: return []
    if isinstance(obj, dict):
        return [f"{k}: {v}" for k, v in obj.items()]
    if isinstance(obj, list):
        return [str(x) for x in obj]
    try:
        j = json.loads(obj)
        if isinstance(j, dict):
            return [f"{k}: {v}" for k, v in j.items()]
    except Exception:
        pass
    return str(obj).splitlines()

def _parse_generic(lines: List[str]):
    actions = []
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        m_id = re.match(r"Agent\s*(\d+)\s*:?", ln, re.I)
        aid = int(m_id.group(1)) if m_id else 0

        if m := _MOVE_RE.search(ln):
            color = m.group(1)
            pos   = (int(m.group(2)), int(m.group(3)))
            direc = m.group(4)
            actions.append((aid, color, pos, direc))

        elif m := _GOAL_RE.search(ln):
            actions.append((aid, m.group(1), None, "goal"))

        elif _NOTH_RE.search(ln):
            actions.append((aid, "none", None, "stay"))

    return actions or None

def _exec_plan(env, plan_like):
    """
    Execute a plan on *env* (plan_like may be a text blob,
    json, dict, or DMAS’s list‑of‑tuples).
    """
    # DMAS already returns list[tuple]
    if isinstance(plan_like, list) and plan_like and len(plan_like[0]) == 4:
        acts = plan_like
    else:
        acts = _parse_generic(_to_lines(plan_like))
        if not acts:
            # fallback to HMAS1 parser
            dummy = HMAS1.HMAS1("boxnet1")
            acts  = HMAS1.HMAS1.parse_llm_plan(dummy, str(plan_like))

    # use HMAS1’s motion executor (it mutates env)
    dummy = HMAS1.HMAS1("boxnet1")
    HMAS1.HMAS1.execute_plan(dummy, env, acts)

# ────────────────────────────────────────────────────────────
#  Wrappers (one per framework)
# ────────────────────────────────────────────────────────────
def wrap_cmas(env):
    prompt        = CMAS.format_prompt(env)
    plan, tokens  = CMAS.call_llm(prompt)
    _exec_plan(env, plan)
    return plan, tokens, 1

def wrap_dmas(env):
    """
    Accepts 2‑tuple (actions, api_calls) or
             3‑tuple (actions, api_calls, tokens)
    from DMAS.dmas_plan.
    Deduplicates actions (keeps most‑recent per agent).
    Filters out any “stay”/“none” actions before execution.
    """
    result = DMAS.dmas_plan(env, env.boxes, env.goals)

    if len(result) == 3:
        actions, api_calls, tokens = result
    else:
        actions, api_calls = result
        tokens = 0

    # keep most‑recent action per agent
    latest = {}
    for act in reversed(actions):
        latest.setdefault(act[0], act)
    unique_actions = list(latest.values())

    # filter out do‑nothing / invalid entries
    exec_actions = [
        a for a in unique_actions
        if a[0] >= 0            # drop parse failures
        and a[3] != "stay"      # drop “do nothing”
        and a[2] is not None    # drop anything w/o a from_pos if you want
    ]

    # now execute only the real moves/goals
    _exec_plan(env, exec_actions)
    return unique_actions, tokens, api_calls

def wrap_hmas1(env):
    etype = "boxnet2" if isinstance(env, BoxNet2) else "boxnet1"
    agent = HMAS1.HMAS1(environment_type=etype)
    agent.env = env
    plan, api_calls = agent.runHMAS1()
    _exec_plan(env, plan)
    return plan, getattr(agent, "token_count", 0), api_calls

def wrap_hmas2(env):
    etype = "boxnet2" if isinstance(env, BoxNet2) else "boxnet1"
    agent = HMAS2.HMAS2(environment_type=etype)
    agent.env = env

    # patch central‐prompt to bind this env
    def fix_prompt():
        h = HMAS1.HMAS1(etype)
        h.env = env
        return h.format_central_prompt(h.env)
    agent.format_central_prompt = fix_prompt

    # count calls
    api_calls = 0
    orig = agent.call_llm
    def patched(prompt):
        nonlocal api_calls
        api_calls += 1
        return orig(prompt)
    agent.call_llm = patched

    plan, _ = agent.runHMAS2()
    _exec_plan(env, plan)
    return plan, getattr(agent, "token_count", 0), api_calls

def wrap_etp(env, max_attempts: int = 3):
    # ensure BoxNet2 agents have `.cell`
    if isinstance(env, BoxNet2):
        for ag in env.agents:
            if not hasattr(ag, "cell"):
                ag.cell = ag.position

    tot_tok = 0
    calls   = 0
    last_actions = None

    for _ in range(max_attempts):
        prompt       = ETP.intialPlan(env)
        reply, used  = ETP.call_llm(prompt)
        tot_tok     += used
        calls       += 1
        acts         = ETP.parse_llm_plan(reply)
        last_actions = acts
        if ETP.execute_plan(env, acts):
            break

    _exec_plan(env, last_actions)
    return last_actions, tot_tok, calls

PLANNERS    = {
    "CMAS"  : wrap_cmas,
    "DMAS"  : wrap_dmas,
    "HMAS‑1": wrap_hmas1,
    "HMAS‑2": wrap_hmas2,
    "ETP"   : wrap_etp,
}
ENVIRONMENTS = {"BoxNet1": BoxNet1, "BoxNet2": BoxNet2}

# ────────────────────────────────────────────────────────────
#  Metric helpers
# ────────────────────────────────────────────────────────────
def success_pct(goal_dict, box_pos_dict, *, is_boxnet2=False):
    """
    • BoxNet1 – normal hit‑rate (boxes on goals).
    • BoxNet2 – ’cleared colors’ fraction (goal list empty).
    """
    if is_boxnet2:
        total   = len(goal_dict)
        cleared = sum(1 for v in goal_dict.values() if len(v) == 0)
        return 100.0 * cleared / total if total else 0.0

    goal_set = {(c, tuple(p)) for c, ps in goal_dict.items() for p in ps}
    box_set  = {(c, tuple(p)) for c, ps in box_pos_dict.items() for p in ps}
    return 100.0 * len(goal_set & box_set) / len(goal_set) if goal_set else 0.0

def step_count(plan):
    if isinstance(plan, str):
        return len([l for l in plan.splitlines() if l.strip()])
    if isinstance(plan, dict):
        return len(plan)
    if isinstance(plan, list):
        return len(plan)
    return 0

# ────────────────────────────────────────────────────────────
#  Batch runner
# ────────────────────────────────────────────────────────────
def batch_test(trials=10, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv = os.path.join(outdir, f"raw_{ts}.csv")
    summ    = os.path.join(outdir, f"summary_{ts}.csv")
    cols    = ["environment","framework","success_rate_pct","steps","api_calls","tokens"]

    rows = []
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for env_name, Env in ENVIRONMENTS.items():
            for fw_name, fw_fn in PLANNERS.items():
                print(f"\n🔹 Framework = {fw_name} / Environment = {env_name}")
                for _ in tqdm(range(trials), desc=f"{fw_name}-{env_name}", unit="trial"):
                    env = Env()
                    try:
                        plan, tokens, calls = fw_fn(env)
                    except Exception:
                        traceback.print_exc()
                        plan, tokens, calls = None, 0, 0

                    row = dict(
                        environment      = env_name,
                        framework        = fw_name,
                        success_rate_pct = success_pct(
                            env.goals,
                            {b.color: list(b.positions) for b in env.boxes},
                            is_boxnet2=isinstance(env, BoxNet2)
                        ),
                        steps     = step_count(plan),
                        api_calls = calls,
                        tokens    = tokens
                    )
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)

    # ── Summary & plots ──────────────────────────────────────
    df  = pd.DataFrame(rows)
    agg = df.groupby(["environment","framework"]).mean(numeric_only=True)
    agg.to_csv(summ)

    for col,label in [
        ("success_rate_pct","Success Rate (%)"),
        ("steps",           "Steps"),
        ("api_calls",       "API Calls"),
        ("tokens",          "Tokens")
    ]:
        plt.figure(figsize=(9,6))
        agg.unstack()[col].plot.bar(title=f"{label} by Framework & Environment")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,f"{col}_{ts}.png"))
        plt.close()

    print("\n✔ Raw CSV   →", raw_csv)
    print("✔ Summary   →", summ)

# ────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n","--trials",type=int,default=5)
    ap.add_argument("-o","--outdir",    default="results")
    args = ap.parse_args()
    batch_test(args.trials,args.outdir)
