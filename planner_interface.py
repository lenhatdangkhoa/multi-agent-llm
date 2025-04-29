from CMAS import format_prompt as cmas_prompt, call_llm as cmas_llm
from DMAS import run_dmas
from HMAS1 import HMAS1
from HMAS2 import HMAS2
from BoxNet2 import BoxNet2


def run_cmas(env):
    prompt = cmas_prompt(env)
    response = cmas_llm(prompt)
    return response.choices[0].message.content, 1, env

def run_dmas_wrapper(env):
    final_plan, total_tokens = run_dmas(env)
    return final_plan, total_tokens, env

def run_hmas1(env):
    env_type = "boxnet2" if isinstance(env, BoxNet2) else "boxnet1"
    planner = HMAS1(environment_type=env_type)
    planner.env = env
    final_plan = planner.run_planning()
    return final_plan, planner.token_count, planner.env

def run_hmas2(env):
    env_type = "boxnet2" if isinstance(env, BoxNet2) else "boxnet1"
    planner = HMAS2(environment_type=env_type)
    planner.env = env
    final_plan = planner.run_planning()
    return final_plan, planner.token_count, planner.env

PLANNERS = {
    # # "CMAS": lambda env: run_cmas(env),
    # # "DMAS": lambda env: run_dmas_wrapper(env),
    # "HMAS-1": lambda env: run_hmas1(env),
    "HMAS-2": lambda env: run_hmas2(env)
}
