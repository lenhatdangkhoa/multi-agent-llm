import os
import time
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import traceback # For detailed error logging
import re # For parsing

# --- Import Environment Models ---
# Ensure these imports match your file structure and the classes used by planners
try:
    from BoxNet1 import BoxNet1
except ImportError:
    print("ERROR: Could not import BoxNet1. Batch testing for BoxNet1 will fail.")
    BoxNet1 = None
try:
    # Using BoxNet2_test based on HMAS code provided earlier [6]
    from BoxNet2_test import BoxNet2
except ImportError:
    print("ERROR: Could not import BoxNet2_test. Batch testing for BoxNet2 will fail.")
    BoxNet2 = None

# --- Import Planner Functions/Classes ---
# Added try-except for robustness if some planner files are missing
try:
    from CMAS import format_prompt as cmas_prompt, call_llm as cmas_llm
except ImportError:
    print("Warning: Could not import CMAS.")
    cmas_prompt, cmas_llm = None, None
try:
    from DMAS import run_dmas
except ImportError:
    print("Warning: Could not import DMAS.")
    run_dmas = None
try:
    from HMAS1 import HMAS1
except ImportError:
    print("Warning: Could not import HMAS1.")
    HMAS1 = None
try:
    from HMAS2 import HMAS2
except ImportError:
    print("Warning: Could not import HMAS2.")
    HMAS2 = None
try:
    # Assuming ETP functions are structured similarly
    from ETP import intialPlan as etp_prompt, call_llm as etp_llm, parse_llm_plan as etp_parse
except ImportError:
    print("Warning: Could not import ETP.")
    etp_prompt, etp_llm, etp_parse = None, None, None


class BatchTester:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.results = defaultdict(lambda: defaultdict(list))
        self.summary = defaultdict(lambda: defaultdict(dict))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"BatchTester initialized. Output directory: {os.path.abspath(output_dir)}")

    # --- Helper for Fallback Plans ---
    def _get_fallback_plan_dict(self, env):
        """Generates a 'do nothing' plan dictionary (for HMAS)."""
        plan = {}
        try:
            num_agents = len(getattr(env, 'agents', []))
            for i in range(num_agents):
                plan[f"Agent{i}"] = "do nothing"
        except Exception as e:
            print(f"Error creating fallback dict plan: {e}")
        return plan

    def _get_fallback_plan_text(self, env):
        """Generates a 'do nothing' plan string (for CMAS, DMAS, ETP)."""
        text = ""
        try:
            num_agents = len(getattr(env, 'agents', []))
            for i in range(num_agents):
                text += f"Agent {i}: do nothing\n"
        except Exception as e:
            print(f"Error creating fallback text plan: {e}")
        return text.strip()

    def _get_fallback_result(self, env, exec_time, tokens=0, api_calls=0, is_json_expected=False, error_msg="Unknown error"):
        """Creates a fallback result dictionary including error flag."""
        if is_json_expected:
            plan = self._get_fallback_plan_dict(env)
        else:
            plan = self._get_fallback_plan_text(env)
        return {
            "plan": plan, "tokens": tokens, "api_calls": api_calls,
            "execution_time": exec_time, "error": True, "error_message": error_msg
        }

    # --- Planner Runner Methods ---
    # Each runner now includes robust try-except and returns fallback on error.
    def run_cmas(self, env):
        if cmas_llm is None: return self._get_fallback_result(env, 0, error_msg="CMAS not imported")
        start_time = time.time()
        tokens, api_calls = 0, 0
        try:
            prompt = cmas_prompt(env)
            # Expecting (content_string, token_count) tuple from CMAS.py
            response_content, token_count = cmas_llm(prompt)
            tokens = token_count
            api_calls = 1 # CMAS is typically one call
            exec_time = time.time() - start_time
            if not isinstance(response_content, str):
                 raise TypeError(f"CMAS returned non-string plan: {type(response_content)}")
            return {"plan": response_content, "tokens": tokens, "api_calls": api_calls, "execution_time": exec_time}
        except Exception as e:
            error_msg = f"Error in CMAS: {e}"
            print(f"{error_msg}\n{traceback.format_exc()}")
            return self._get_fallback_result(env, time.time() - start_time, tokens, api_calls, is_json_expected=False, error_msg=error_msg)

    def run_dmas(self, env):
        if run_dmas is None: return self._get_fallback_result(env, 0, error_msg="DMAS not imported")
        start_time = time.time()
        tokens, api_calls = 0, 0
        try:
            # Expecting (plan_text_string, token_count) tuple from DMAS.py
            plan_text, token_count = run_dmas(env)
            tokens = token_count
            # Approximation for API calls in DMAS
            api_calls = max(1, tokens // 1000) if tokens > 0 else 0
            exec_time = time.time() - start_time
            if not isinstance(plan_text, str):
                 raise TypeError(f"DMAS returned non-string plan: {type(plan_text)}")
            return {"plan": plan_text, "tokens": tokens, "api_calls": api_calls, "execution_time": exec_time}
        except Exception as e:
            error_msg = f"Error in DMAS: {e}"
            print(f"{error_msg}\n{traceback.format_exc()}")
            return self._get_fallback_result(env, time.time() - start_time, tokens, api_calls, is_json_expected=False, error_msg=error_msg)

    def run_hmas1(self, env):
        if HMAS1 is None: return self._get_fallback_result(env, 0, error_msg="HMAS1 not imported", is_json_expected=True)
        start_time = time.time()
        tokens, api_calls = 0, 0
        planner = None
        try:
            env_type = "boxnet1" if isinstance(env, BoxNet1) else "boxnet2"
            planner = HMAS1(environment_type=env_type)
            planner.env = env
            # run_planning returns dict on consensus, None otherwise
            plan_result = planner.run_planning() # This is a dict or None
            tokens = getattr(planner, 'token_count', 0)
            # Estimate API calls - needs refinement based on HMAS1 logic (e.g., rounds * agents)
            api_calls = max(1, tokens // 1000) if tokens > 0 else 0

            if plan_result is None:
                print(f"HMAS1 returned None (no consensus). Using fallback dict plan.")
                plan = self._get_fallback_plan_dict(env) # HMAS expects JSON dict
                fallback_used = True
            else:
                if not isinstance(plan_result, dict):
                    raise TypeError(f"HMAS1 returned non-dict/None plan: {type(plan_result)}")
                plan = plan_result
                fallback_used = False

            exec_time = time.time() - start_time
            result = {"plan": plan, "tokens": tokens, "api_calls": api_calls, "execution_time": exec_time}
            if fallback_used: result["error"] = True; result["error_message"] = "HMAS1 No Consensus" # Mark fallback
            return result
        except Exception as e:
            error_msg = f"Error in HMAS1: {e}"
            print(f"{error_msg}\n{traceback.format_exc()}")
            if planner: tokens = getattr(planner, 'token_count', 0) # Try to get tokens even on error
            return self._get_fallback_result(env, time.time() - start_time, tokens, api_calls, is_json_expected=True, error_msg=error_msg)

    def run_hmas2(self, env):
        if HMAS2 is None: return self._get_fallback_result(env, 0, error_msg="HMAS2 not imported", is_json_expected=True)
        start_time = time.time()
        tokens, api_calls = 0, 0
        planner = None
        try:
            env_type = "boxnet1" if isinstance(env, BoxNet1) else "boxnet2"
            planner = HMAS2(environment_type=env_type)
            planner.env = env
            # run_planning returns dict (best attempt) or None [user code update]
            plan_result = planner.run_planning() # This is a dict or None
            tokens = getattr(planner, 'token_count', 0)
            # Estimate API calls - needs refinement based on HMAS2 logic
            api_calls = max(1, tokens // 1000) if tokens > 0 else 0

            if plan_result is None:
                # HMAS2's run_planning was updated to return best_plan or fallback dict, so None should be rare unless error occurred before return
                print(f"HMAS2 returned None unexpectedly. Using fallback dict plan.")
                plan = self._get_fallback_plan_dict(env) # HMAS expects JSON dict
                fallback_used = True
                error_msg = "HMAS2 returned None"
            else:
                 if not isinstance(plan_result, dict):
                     raise TypeError(f"HMAS2 returned non-dict plan: {type(plan_result)}")
                 plan = plan_result
                 # Check if the returned plan is the fallback created *within* HMAS2 (less ideal)
                 is_internal_fallback = all(v.lower() == "do nothing" for v in plan.values())
                 if is_internal_fallback and len(plan) == len(getattr(env, 'agents', [])):
                      print("HMAS2 returned internal 'do nothing' fallback plan.")
                      # Decide if this counts as an error/no consensus for reporting
                      fallback_used = True # Treat internal fallback as non-success origin
                      error_msg = "HMAS2 Internal Fallback"
                 else:
                      fallback_used = False

            exec_time = time.time() - start_time
            result = {"plan": plan, "tokens": tokens, "api_calls": api_calls, "execution_time": exec_time}
            if fallback_used: result["error"] = True; result["error_message"] = error_msg
            return result
        except Exception as e:
            error_msg = f"Error in HMAS2: {e}"
            print(f"{error_msg}\n{traceback.format_exc()}")
            if planner: tokens = getattr(planner, 'token_count', 0)
            return self._get_fallback_result(env, time.time() - start_time, tokens, api_calls, is_json_expected=True, error_msg=error_msg)

    def run_etp(self, env):
        if etp_llm is None or etp_prompt is None or etp_parse is None:
            return self._get_fallback_result(env, 0, error_msg="ETP not imported")
        start_time = time.time()
        total_tokens = 0
        api_calls = 0
        planning_attempts = 0
        max_attempts = 5 # Limit replanning
        final_plan_text = None
        plan_generated_successfully = False # Track if any valid plan attempt completed

        try:
            # ETP needs a modifiable env state for replanning
            if isinstance(env, BoxNet1) and BoxNet1:
                planning_env = BoxNet1() # Assumes __init__ resets state
            elif isinstance(env, BoxNet2) and BoxNet2:
                 planning_env = BoxNet2() # Assumes __init__ resets state
            else:
                 raise TypeError(f"Unsupported environment type for ETP copy: {type(env)}")
            # TODO: Ensure planning_env state matches initial 'env' state if __init__ doesn't guarantee it

            while planning_attempts < max_attempts:
                planning_attempts += 1
                current_api_calls = 1 # Assume 1 call per attempt
                api_calls += current_api_calls

                prompt = etp_prompt(planning_env)
                # Expecting content string from ETP.py call_llm
                current_plan_text = etp_llm(prompt)
                if not isinstance(current_plan_text, str):
                     print(f"ETP Warning: Attempt {planning_attempts} returned non-string plan. Skipping attempt.")
                     continue # Try next attempt

                # Approx token count for this attempt
                prompt_tokens = len(prompt.split())
                response_tokens = len(current_plan_text.split())
                total_tokens += prompt_tokens + response_tokens

                final_plan_text = current_plan_text # Store the latest plan text

                # Parse and Validate silently using a *fresh* copy for *this attempt*
                actions = etp_parse(current_plan_text) # Use ETP's parser
                if not actions:
                     print(f"ETP Warning: Attempt {planning_attempts} plan parsed to empty actions. Assuming failure.")
                     # Optionally: Update planning_env state based on failure if ETP logic requires
                     continue # Try next attempt

                # Create a fresh validation env based on the *current* state of planning_env
                # This is tricky. If ETP is meant to learn from failed *execution*,
                # the validation needs to simulate that failure's effect on planning_env.
                # Simpler: Validate if the plan *would* work from the planning_env's current state.
                temp_validation_env = type(planning_env)() # Fresh instance
                # TODO: Make temp_validation_env state match planning_env's *current* state
                # This might require a copy method on the environment classes.
                # Assuming for now __init__ is sufficient and we validate from initial state.
                # If ETP state matters, validation needs `planning_env` state.
                # success_this_attempt = self.validate_plan(planning_env, current_plan_text)[0] # Validate against current state?
                # Safest for now: validate against initial state like other planners
                validation_env_initial = type(env)()
                success_this_attempt = self.validate_plan(validation_env_initial, current_plan_text)[0]


                if success_this_attempt:
                    print(f"ETP succeeded on attempt {planning_attempts}")
                    plan_generated_successfully = True
                    break # Exit loop on success
                else:
                    print(f"ETP attempt {planning_attempts} failed validation. Replanning...")
                    # How does ETP update planning_env state for the next prompt?
                    # This needs to be handled correctly in ETP.py or by modifying planning_env here
                    # based on the failed validation, which is complex.
                    # Assuming etp_prompt(planning_env) gets the right state somehow.

            exec_time = time.time() - start_time

            if not plan_generated_successfully: # Handle case where loop finished without success
                 print("ETP failed to generate a successful plan within max attempts.")
                 if final_plan_text is None: # Also handle case where loop didn't even run once
                      final_plan_text = self._get_fallback_plan_text(env)
                 return self._get_fallback_result(env, exec_time, total_tokens, api_calls, is_json_expected=False, error_msg="ETP Max Attempts Reached or No Valid Plan")

            return {"plan": final_plan_text, "tokens": total_tokens, "api_calls": api_calls, "execution_time": exec_time}

        except Exception as e:
            error_msg = f"Error in ETP: {e}"
            print(f"{error_msg}\n{traceback.format_exc()}")
            return self._get_fallback_result(env, time.time() - start_time, total_tokens, api_calls, is_json_expected=False, error_msg=error_msg)


    # --- Validation and Parsing Methods ---

    def parse_llm_plan(self, text_or_dict):
        """Parse LLM output (string or dict) into actionable steps [(agent_id, color, from_pos, direction)]."""
        actions = []
        if text_or_dict is None:
             # print("Parse Debug: Received None plan.")
             return actions # Return empty list for None input

        lines = []
        plan_type = "Unknown"
        if isinstance(text_or_dict, dict):
            # Handle HMAS JSON output
            lines = [f"Agent {k.replace('Agent','')}: {v}" for k, v in text_or_dict.items()]
            plan_type = "dict"
        elif isinstance(text_or_dict, str):
            # Handle CMAS/DMAS/ETP string output
            lines = text_or_dict.strip().split('\n')
            # Filter potentially empty lines and common preamble/postamble noise
            lines = [line for line in lines if line.strip() and not line.strip().lower().startswith("plan:") and not line.strip().lower().startswith("here is")]
            plan_type = "str"
        else:
            print(f"Parse Error: Unexpected plan type {type(text_or_dict)}. Content: {text_or_dict}")
            return actions

        # print(f"Parse Debug: Input type={plan_type}, Num lines={len(lines)}")

        # --- Regex patterns ---
        dir_map = {"north": "up", "south": "down", "east": "right", "west": "left"}
        # Added ^ anchor, flexible whitespace \s+, optional -?, ignore case
        pattern_move = r"^\s*-?\s?Agent\s+(\d+)\s*:\s*move\s+(\w+)\s+box\s+from\s+\((\d+)\s*,\s*(\d+)\)\s+to\s+\((\d+)\s*,\s*(\d+)\)\s*(?:\[?(\w+)\]?)?"
        pattern_nothing = r"^\s*-?\s?Agent\s+(\d+)\s*:\s*do\s+nothing"
        pattern_standby = r"^\s*-?\s?Agent\s+(\d+)\s*:\s*[Ss]tandby"
        pattern_move_to_goal = r"^\s*-?\s?Agent\s+(\d+)\s*:\s*move\s+(\w+)\s+box\s+to\s+goal"
        # --- End Regex ---

        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            # print(f"Parse Debug: Processing line {i+1}: {line}")

            action_matched = False
            move_match = re.match(pattern_move, line, re.IGNORECASE)
            if move_match:
                action_matched = True
                agent_id = int(move_match.group(1))
                color = move_match.group(2).lower()
                from_pos = (int(move_match.group(3)), int(move_match.group(4)))
                raw_dir = move_match.group(7).lower() if move_match.group(7) else ""
                direction = dir_map.get(raw_dir, raw_dir)
                if direction not in ["up", "down", "left", "right"]:
                     # print(f"Parse Warning: Unknown direction '{direction}' mapped to '{raw_dir}' in line: {line}")
                     direction = "stay" # Default to 'stay' or skip? Let's use 'stay'
                actions.append((agent_id, color, from_pos, direction))
                # print(f"Parse Debug: Matched MOVE - Action: {(agent_id, color, from_pos, direction)}")
                continue # Go to next line

            nothing_match = re.match(pattern_nothing, line, re.IGNORECASE)
            if nothing_match:
                action_matched = True
                agent_id = int(nothing_match.group(1))
                actions.append((agent_id, "none", None, "stay"))
                # print(f"Parse Debug: Matched NOTHING - Action: {(agent_id, 'none', None, 'stay')}")
                continue

            standby_match = re.match(pattern_standby, line, re.IGNORECASE)
            if standby_match:
                action_matched = True
                agent_id = int(standby_match.group(1))
                actions.append((agent_id, "none", None, "stay")) # Treat standby as do nothing
                # print(f"Parse Debug: Matched STANDBY - Action: {(agent_id, 'none', None, 'stay')}")
                continue

            move_to_goal_match = re.match(pattern_move_to_goal, line, re.IGNORECASE)
            if move_to_goal_match:
                action_matched = True
                agent_id = int(move_to_goal_match.group(1))
                color = move_to_goal_match.group(2).lower()
                # For 'move to goal', from_pos is not specified in the command, so use None
                actions.append((agent_id, color, None, "goal"))
                # print(f"Parse Debug: Matched MOVE_TO_GOAL - Action: {(agent_id, color, None, 'goal')}")
                continue

            # If no pattern matched this line
            if not action_matched:
                 print(f"Parse Warning: Unrecognized plan line skipped: {line}")

        # print(f"Parse Debug: Total actions parsed: {len(actions)}")
        return actions

    def validate_plan(self, env_instance, plan_obj):
        """
        Validate if a plan successfully completes the task using a fresh environment instance.
        Returns (bool: success, int: steps_executed_or_parsed).
        """
        # print(f"\n--- Starting Validation for Env: {type(env_instance).__name__} ---")
        validation_env = None
        try:
            # Create a fresh copy for validation simulation
            # Relies on env __init__ providing a clean, identical starting state
            if isinstance(env_instance, BoxNet1) and BoxNet1:
                validation_env = BoxNet1()
            elif isinstance(env_instance, BoxNet2) and BoxNet2:
                 validation_env = BoxNet2()
            else:
                 print(f"Validation Error: Unknown env type {type(env_instance)}")
                 return False, 0

            actions = self.parse_llm_plan(plan_obj)
            parsed_steps = len(actions) # Number of actions parsed

            # --- Execute Silently on validation_env ---
            execution_successful = True # Assume success unless a move fails
            executed_step_count = 0

            if not actions:
                 execution_successful = True # No moves failed, check initial state goals below
                 # print("Validation Info: Parsed plan has no actions.")
            else:
                for step_idx, (agent_id, color, from_pos, direction) in enumerate(actions):
                    executed_step_count = step_idx + 1 # Track how many steps were attempted
                    # print(f"Validating Step {executed_step_count}: {(agent_id, color, from_pos, direction)}")

                    if color == "none": continue # Skip 'do nothing' / 'stay'

                    if direction == "goal":
                        # Check validity based on environment type
                        if isinstance(validation_env, BoxNet1):
                             print(f"Validation Fail (Step {executed_step_count}): 'move_to_goal' action invalid for BoxNet1.")
                             execution_successful = False
                             break
                        if not isinstance(validation_env, BoxNet2):
                             print(f"Validation Fail (Step {executed_step_count}): 'move_to_goal' action logic not defined for {type(validation_env)}.")
                             execution_successful = False
                             break

                        # Logic for BoxNet2 move_to_goal
                        if color not in validation_env.goals or not validation_env.goals.get(color): # Use .get for safety
                             print(f"Validation Fail (Step {executed_step_count}): Goal for {color} already met or doesn't exist in validation env. Goals: {validation_env.goals}")
                             execution_successful = False
                             break
                        # We need the box's current position to remove it correctly in BoxNet2
                        current_box = next((b for b in validation_env.boxes if b.color == color), None)
                        if not current_box:
                             print(f"Validation Fail (Step {executed_step_count}): Box {color} not found in validation env for move_to_goal.")
                             execution_successful = False
                             break
                        # BoxNet2 move_to_goal modifies internal state
                        validation_env.move_to_goal(color) # Assumes this works correctly
                        # print(f"Validation OK (Step {executed_step_count}): Moved {color} to goal.")
                        continue

                    # --- Handle standard 'move' action ---
                    if from_pos is None:
                        print(f"Validation Fail (Step {executed_step_count}): 'move' action requires 'from_pos', but got None.")
                        execution_successful = False
                        break

                    # Find box in the validation_env at the specified from_pos
                    box_found = False
                    target_box = None
                    for box in validation_env.boxes:
                         if box.color == color:
                              current_positions = getattr(box, 'positions', []) # BoxNet1 uses list
                              if from_pos in current_positions:
                                   box_found = True
                                   target_box = box
                                   break
                    if not box_found:
                        print(f"Validation Fail (Step {executed_step_count}): Box {color} not found at expected position {from_pos}. Current boxes: {[(b.color, getattr(b, 'positions', [])) for b in validation_env.boxes]}")
                        execution_successful = False
                        break

                    # Attempt move in validation_env
                    move_success = validation_env.move_box(target_box, from_pos, direction) # Modifies box.positions
                    if not move_success:
                        print(f"Validation Fail (Step {executed_step_count}): env.move_box failed for {color} from {from_pos} {direction}.")
                        execution_successful = False
                        break
                    # print(f"Validation OK (Step {executed_step_count}): Moved {color} via move_box.")
            # --- End Execute Silently Loop ---

            # If any step failed during execution, validation fails
            if not execution_successful:
                 # print("--- Validation Result: Execution Failed ---")
                 return False, executed_step_count # Return steps attempted before failure

            # --- Final Goal State Check (Environment Specific) ---
            # This check runs only if all execution steps were successful
            all_goals_met = False
            if isinstance(validation_env, BoxNet1):
                goals_met_count = 0
                required_goals = len([b for b in validation_env.boxes if validation_env.goals.get(b.color)]) # Count boxes that HAVE a goal defined
                if required_goals == 0 and not validation_env.boxes: # Handle edge case: no boxes/goals initially
                    all_goals_met = True
                else:
                    for box in validation_env.boxes:
                        target_goal_positions = validation_env.goals.get(box.color, [])
                        if not target_goal_positions: continue # Skip boxes without goals

                        current_positions = getattr(box, 'positions', [])
                        # Check if *any* current position matches *any* target goal position
                        if any(pos in target_goal_positions for pos in current_positions):
                            goals_met_count += 1
                    all_goals_met = (goals_met_count == required_goals)
                # print(f"BoxNet1 Final Goal Check: Goals Met Count={goals_met_count}, Required={required_goals}, Result={all_goals_met}")
                # print(f"Final Box Positions: {[(b.color, getattr(b, 'positions', [])) for b in validation_env.boxes]}")

            elif isinstance(validation_env, BoxNet2):
                # BoxNet2 uses the goal list emptying mechanism via move_to_goal
                all_goals_met = all(not goal_list for goal_list in validation_env.goals.values())
                # print(f"BoxNet2 Final Goal Check: Empty Goal Lists Check Result={all_goals_met}, Final Goals Dict={validation_env.goals}")

            # print(f"--- Validation Result: {'Success' if all_goals_met else 'Failure'} ---")
            return all_goals_met, executed_step_count # Return success status and steps executed

        except Exception as e:
            print(f"CRITICAL Error during plan validation: {e}\n{traceback.format_exc()}")
            # print("--- Validation Result: Error ---")
            return False, 0 # Treat validation errors as failures


    # --- Main Test Loop ---
    def run_tests(self, num_trials=10, agent_counts=[4]): # Default to 4 agents for quicker debug
        """Run tests for all planners on all environments"""
        # Select planners that were successfully imported
        planners = {}
        if cmas_llm: planners["CMAS"] = self.run_cmas
        if run_dmas: planners["DMAS"] = self.run_dmas
        if HMAS1: planners["HMAS-1"] = self.run_hmas1
        if HMAS2: planners["HMAS-2"] = self.run_hmas2
        if etp_llm: planners["ETP"] = self.run_etp

        if not planners:
            print("Error: No planners could be imported. Exiting.")
            return {}

        environments = {}
        if BoxNet1: environments["BoxNet1"] = BoxNet1
        if BoxNet2: environments["BoxNet2"] = BoxNet2 # Use the imported class

        if not environments:
            print("Error: No environments could be imported. Exiting.")
            return {}

        # Run tests
        for env_name, env_class in environments.items():
            print(f"\n=== Testing on {env_name} ===")
            for agent_count in agent_counts:
                # TODO: Adapt agent count if environments support it.
                print(f"\n-- Testing with {agent_count} agents (Note: Environment might use its default agent setup) --")

                for planner_name, planner_fn in planners.items():
                    print(f"Running {planner_name}...")
                    success_count, total_tokens, total_api_calls, total_steps, total_time = 0, 0, 0, 0, 0.0

                    for trial in tqdm(range(num_trials), desc=f"{planner_name} ({agent_count} agents)", unit="trial"):
                        env_instance_for_planner = None # Define for potential error logging
                        try:
                            # Create a fresh environment instance for the planner to see
                            env_instance_for_planner = env_class()
                            # TODO: Pass agent_count if supported: env_instance_for_planner = env_class(num_agents=agent_count)
                        except Exception as e:
                             print(f"\nError creating environment {env_name} for trial {trial+1}: {e}. Skipping trial.")
                             # Log minimal error for this trial
                             self.results[env_name][planner_name].append({
                                 "agent_count": agent_count, "trial": trial + 1, "success": False, "error": f"Env Creation Failed: {e}"
                             })
                             continue # Skip to next trial

                        try:
                            # Run the planner function
                            result = planner_fn(env_instance_for_planner)
                            plan_obj = result.get("plan") # Can be str or dict or None
                            tokens = result.get("tokens", 0)
                            api_calls = result.get("api_calls", 0)
                            execution_time = result.get("execution_time", 0.0)
                            planner_had_error = result.get("error", False) # Did the runner use fallback?
                            error_message = result.get("error_message", "")

                            trial_success = False
                            steps_executed = 0

                            if plan_obj is None or planner_had_error:
                                # print(f"Trial {trial+1}: Planner failed or returned no plan/fallback. Plan Obj: {plan_obj}, Error Flag: {planner_had_error}, Msg: {error_message}")
                                trial_success = False
                                steps_executed = 0
                            else:
                                # Validate the plan using a separate fresh environment instance
                                validation_env_instance = env_class()
                                # TODO: Match agent count if needed
                                validation_success, steps = self.validate_plan(validation_env_instance, plan_obj)
                                trial_success = validation_success
                                steps_executed = steps
                                # print(f"Trial {trial+1}: Validation Result Success={trial_success}, Steps Executed/Attempted={steps_executed}")

                            # Log detailed results for this trial
                            self.results[env_name][planner_name].append({
                                "agent_count": agent_count, "trial": trial + 1,
                                "success": trial_success,
                                "steps_executed": steps_executed if trial_success else steps_executed, # Log steps even on failure
                                "tokens": tokens, "api_calls": api_calls,
                                "execution_time": execution_time,
                                "planner_error": planner_had_error,
                                "planner_error_message": error_message,
                                # Log plan safely - convert dict to string for consistency if needed, handle None
                                "plan": json.dumps(plan_obj) if isinstance(plan_obj, dict) else str(plan_obj) if plan_obj is not None else "None"
                            })

                            # Aggregate metrics only if validation successful
                            if trial_success:
                                success_count += 1
                                total_tokens += tokens
                                total_api_calls += api_calls
                                total_steps += steps_executed # Use steps executed from validation
                                total_time += execution_time

                        except Exception as e:
                            # Catch unexpected errors during the trial processing itself
                            print(f"\nCRITICAL Error during trial {trial+1} for {planner_name}: {e}\n{traceback.format_exc()}")
                            # Log error for this trial
                            self.results[env_name][planner_name].append({
                                "agent_count": agent_count, "trial": trial + 1, "success": False, "error": f"Trial Processing Error: {e}"
                            })
                            # Continue to next trial

                    # --- Calculate and store summary metrics for this planner/agent_count ---
                    success_rate = success_count / num_trials if num_trials > 0 else 0
                    # Averages are calculated based on SUCCESSFUL trials only
                    avg_tokens = total_tokens / success_count if success_count > 0 else 0
                    avg_api_calls = total_api_calls / success_count if success_count > 0 else 0
                    avg_steps = total_steps / success_count if success_count > 0 else 0
                    avg_time = total_time / success_count if success_count > 0 else 0

                    self.summary[env_name][planner_name][agent_count] = {
                        "success_rate": success_rate, "avg_tokens_success": avg_tokens,
                        "avg_api_calls_success": avg_api_calls, "avg_steps_success": avg_steps,
                        "avg_time_success": avg_time,
                        "trials": num_trials, "successful_trials": success_count
                    }

                    print(f"\n{planner_name} results with {agent_count} agents ({num_trials} trials):")
                    print(f"  Success rate: {success_rate:.2%}")
                    print(f"  Avg tokens (successful): {avg_tokens:.2f}")
                    print(f"  Avg API calls (successful): {avg_api_calls:.2f}")
                    print(f"  Avg steps (successful): {avg_steps:.2f}")
                    print(f"  Avg execution time (successful): {avg_time:.2f}s")

        # --- Save results and generate plots ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_path = os.path.join(self.output_dir, f"detailed_results_{timestamp}.json")
        summary_path = os.path.join(self.output_dir, f"summary_results_{timestamp}.json")

        try:
            # Use default=str to handle potential non-serializable types if any sneak in
            with open(detailed_path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            with open(summary_path, "w") as f:
                json.dump(self.summary, f, indent=2, default=str)
            print(f"\nResults saved to {detailed_path} and {summary_path}")
        except Exception as e:
            print(f"\nError saving results: {e}")

        try:
            self.generate_plots(timestamp)
            print(f"Plots generated in {os.path.join(self.output_dir, f'plots_{timestamp}')}")
        except Exception as e:
            print(f"\nError generating plots: {e}")

        return self.summary

    # --- Plotting Method (Modified for new metric names) ---
    def generate_plots(self, timestamp):
        """Generate plots from test results"""
        plots_dir = os.path.join(self.output_dir, f"plots_{timestamp}")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Updated metric keys from summary dict
        metrics = ["success_rate", "avg_tokens_success", "avg_api_calls_success", "avg_steps_success", "avg_time_success"]
        titles = ["Success Rate", "Avg Tokens (Successful)", "Avg API Calls (Successful)", "Avg Steps (Successful)", "Avg Execution Time (Successful)"]

        for metric, title in zip(metrics, titles):
            try:
                self._plot_metric(metric, title, plots_dir)
            except Exception as e:
                 print(f"Error plotting {metric}: {e}")

    def _plot_metric(self, metric, title, plots_dir):
        """Helper method to plot a specific metric using summary data."""
        for env_name in self.summary.keys():
            plt.figure(figsize=(10, 6))
            has_data = False
            agent_counts_tested = set() # Collect all agent counts tested for this env

            for planner_name in self.summary[env_name].keys():
                 if not self.summary[env_name][planner_name]: continue # Skip if no data for this planner

                 agent_counts = sorted(self.summary[env_name][planner_name].keys())
                 if not agent_counts: continue # Skip if no agent counts tested for this planner

                 agent_counts_tested.update(agent_counts) # Add to overall set for x-axis ticks
                 values = [self.summary[env_name][planner_name][ac].get(metric, 0) for ac in agent_counts] # Use .get for safety

                 plt.plot(agent_counts, values, marker='o', linestyle='-', label=planner_name)
                 has_data = True

            if not has_data:
                 plt.close() # Close figure if no data plotted for this environment
                 continue

            plt.title(f"{title} - {env_name}")
            plt.xlabel("Number of Agents")
            plt.ylabel(title)
            # Ensure x-axis ticks cover all tested agent counts, even if some planners missed counts
            sorted_agent_ticks = sorted(list(agent_counts_tested))
            if sorted_agent_ticks: plt.xticks(sorted_agent_ticks)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.ylim(bottom=0) # Ensure plots start at 0
            if "rate" in metric: plt.ylim(top=min(1.05, max(1.0, plt.ylim()[1]))) # Cap rate plots near 100% unless data exceeds it

            plot_path = os.path.join(plots_dir, f"{env_name}_{metric}.png")
            try:
                 plt.savefig(plot_path)
            except Exception as e:
                 print(f"Error saving plot {plot_path}: {e}")
            plt.close() # Close figure after saving


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch testing framework for multi-robot planning")
    # Changed defaults for faster debugging
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per configuration")
    parser.add_argument("--agents", type=int, nargs="+", default=[4], help="Agent counts to test (e.g., 4 8)")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()

    if not BoxNet1 and not BoxNet2:
        print("CRITICAL ERROR: Neither BoxNet1 nor BoxNet2 environments could be imported. Exiting.")
    elif not args.agents:
        print("Error: No agent counts specified via --agents argument.")
    else:
        print(f"Starting Batch Test: Trials={args.trials}, Agents={args.agents}, Output='{args.output}'")
        tester = BatchTester(output_dir=args.output)
        summary = tester.run_tests(num_trials=args.trials, agent_counts=args.agents)
        print("\n=== Testing Complete ===")
        if summary:
             print("Summary results generated.")
        else:
             print("Testing finished, but no summary results were generated (check logs for errors).")
