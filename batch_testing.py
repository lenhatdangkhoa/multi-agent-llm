import os
import time
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Import environment models
from BoxNet1 import BoxNet1
from BoxNet2_test import BoxNet2

# Import planners
from CMAS import format_prompt as cmas_prompt, call_llm as cmas_llm
from DMAS import run_dmas
from HMAS1 import HMAS1
from HMAS2 import HMAS2
from ETP import intialPlan, call_llm as etp_llm, parse_llm_plan


class BatchTester:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.results = defaultdict(lambda: defaultdict(list))
        self.summary = defaultdict(lambda: defaultdict(dict))

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run_cmas(self, env):
        """Run CMAS planner and return metrics"""
        start_time = time.time()
        prompt = cmas_prompt(env)
        response_content, token_count = cmas_llm(prompt)  # Unpack the tuple here
        api_calls = 1  # CMAS uses a single API call
        execution_time = time.time() - start_time

        return {
            "plan": response_content,  # Use the content directly
            "tokens": token_count,
            "api_calls": api_calls,
            "execution_time": execution_time
        }

    def run_dmas(self, env):
        """Run DMAS planner and return metrics"""
        start_time = time.time()
        plan_text, token_count = run_dmas(env)
        # DMAS typically uses multiple API calls
        api_calls = token_count // 1000  # Approximate based on token usage
        execution_time = time.time() - start_time

        return {
            "plan": plan_text,
            "tokens": token_count,
            "api_calls": api_calls,
            "execution_time": execution_time
        }

    def run_hmas1(self, env):
        """Run HMAS-1 planner and return metrics"""
        start_time = time.time()
        env_type = "boxnet1" if isinstance(env, BoxNet1) else "boxnet2"
        planner = HMAS1(environment_type=env_type)
        planner.env = env
        plan = planner.run_planning()
        execution_time = time.time() - start_time

        return {
            "plan": plan,
            "tokens": planner.token_count,
            "api_calls": planner.token_count // 1000,  # Approximate
            "execution_time": execution_time
        }

    def run_hmas2(self, env):
        """Run HMAS-2 planner and return metrics"""
        start_time = time.time()
        env_type = "boxnet1" if isinstance(env, BoxNet1) else "boxnet2"
        planner = HMAS2(environment_type=env_type)
        planner.env = env
        plan = planner.run_planning()
        execution_time = time.time() - start_time

        return {
            "plan": plan,
            "tokens": planner.token_count,
            "api_calls": planner.token_count // 1000,  # Approximate
            "execution_time": execution_time
        }

    def run_etp(self, env):
        """Run ETP planner and return metrics"""
        start_time = time.time()

        total_tokens = 0
        api_calls = 0
        planning_attempts = 0

        # Create a copy of the environment for planning
        if isinstance(env, BoxNet1):
            planning_env = BoxNet1()
        else:
            planning_env = BoxNet2()

        # Keep replanning until successful
        success = False
        plan_text = None

        while not success and planning_attempts < 5:  # Limit attempts to prevent infinite loops
            planning_attempts += 1
            api_calls += 1

            prompt = intialPlan(planning_env)
            plan_text = etp_llm(prompt)
            total_tokens += len(prompt.split()) + len(plan_text.split())  # Approximate token count

            actions = parse_llm_plan(plan_text)
            success = self.execute_plan_silently(planning_env, actions)

        execution_time = time.time() - start_time

        return {
            "plan": plan_text,
            "tokens": total_tokens,
            "api_calls": api_calls,
            "execution_time": execution_time
        }

    def execute_plan_silently(self, env, actions):
        """Execute the plan without visual output to check validity."""
        for agent_id, color, from_pos, direction in actions:
            if color == "none":
                continue

            if direction == "goal":
                env.move_to_goal(color)
                continue

            # Find the box object
            box = next((b for b in env.boxes if b.color == color and from_pos in b.positions), None)
            if not box:
                return False

            success = env.move_box(box, from_pos, direction)
            if not success:
                return False

        return True

    def validate_plan(self, env, plan_text):
        """Validate if a plan successfully completes the task"""
        # Create a copy of the environment for validation
        if isinstance(env, BoxNet1):
            test_env = BoxNet1()
        else:
            test_env = BoxNet2()

        # Parse the plan
        actions = self.parse_llm_plan(plan_text)

        # Execute each action in the plan
        steps = 0
        for agent_id, color, from_pos, direction in actions:
            steps += 1
            if color == "none":
                continue  # Agent does nothing

            if direction == "goal":
                test_env.move_to_goal(color)
                continue

            # Find the box
            box = next((b for b in test_env.boxes if b.color == color and from_pos in b.positions), None)
            if not box:
                return False, steps  # Box not found, plan failed

            # Move the box
            success = test_env.move_box(box, from_pos, direction)
            if not success:
                return False, steps  # Move failed, plan failed

        # Check if all boxes are at their goals
        for color, goals in test_env.goals.items():
            if goals:  # If there are still goals for this color, not all boxes are at goals
                return False, steps

        return True, steps  # All boxes are at their goals, plan succeeded

    def parse_llm_plan(self, text):
        """Parse LLM output into actionable steps"""
        actions = []

        if isinstance(text, dict):
            lines = [f"Agent {k.replace('Agent', '')}: {v}" for k, v in text.items()]
        else:
            lines = text.strip().split('\n')
            lines = [line for line in lines if line.strip() and line.strip().lower() != "plan:"]

        # Direction mapping
        dir_map = {"north": "up", "south": "down", "east": "right", "west": "left"}

        # Regex patterns for different action formats
        import re
        pattern_move = r"-? ?Agent (\d+): move (\w+) box from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)\s*(?:\[?(\w+)\]?)?"
        pattern_nothing = r"-? ?Agent (\d+): do nothing"
        pattern_move_to_goal = r"-? ?Agent (\d+): move (\w+) box to goal"
        pattern_standby = r"-? ?Agent (\d+): [Ss]tandby"

        for line in lines:
            line = line.strip()
            move_match = re.match(pattern_move, line)
            nothing_match = re.match(pattern_nothing, line)
            move_to_goal_match = re.match(pattern_move_to_goal, line)
            standby_match = re.match(pattern_standby, line)

            if move_match:
                agent_id = int(move_match.group(1))
                color = move_match.group(2)
                from_pos = (int(move_match.group(3)), int(move_match.group(4)))
                raw_dir = move_match.group(7).lower() if move_match.group(7) else ""
                direction = dir_map.get(raw_dir, raw_dir)
                actions.append((agent_id, color, from_pos, direction))
            elif nothing_match or standby_match:
                agent_id = int((nothing_match or standby_match).group(1))
                actions.append((agent_id, "none", None, "stay"))
            elif move_to_goal_match:
                agent_id = int(move_to_goal_match.group(1))
                color = move_to_goal_match.group(2)
                actions.append((agent_id, color, None, "goal"))

        return actions

    def run_tests(self, num_trials=10, agent_counts=[4, 8, 16, 32]):
        """Run tests for all planners on all environments"""
        planners = {
            "CMAS": self.run_cmas,
            "DMAS": self.run_dmas,
            "HMAS-1": self.run_hmas1,
            "HMAS-2": self.run_hmas2,
            "ETP": self.run_etp
        }

        environments = {
            "BoxNet1": BoxNet1,
            "BoxNet2": BoxNet2
        }

        # Run tests for each environment, planner, and agent count
        for env_name, env_class in environments.items():
            print(f"\n=== Testing on {env_name} ===")

            for agent_count in agent_counts:
                print(f"\n-- Testing with {agent_count} agents --")

                for planner_name, planner_fn in planners.items():
                    print(f"Running {planner_name}...")

                    success_count = 0
                    total_tokens = 0
                    total_api_calls = 0
                    total_steps = 0
                    total_time = 0

                    for trial in tqdm(range(num_trials)):
                        # Create environment with specified agent count
                        env = env_class()

                        # Run planner
                        try:
                            result = planner_fn(env)
                            plan = result["plan"]
                            tokens = result["tokens"]
                            api_calls = result["api_calls"]
                            execution_time = result["execution_time"]

                            # Validate plan
                            success, steps = self.validate_plan(env, plan)

                            # Record metrics
                            if success:
                                success_count += 1
                                total_tokens += tokens
                                total_api_calls += api_calls
                                total_steps += steps
                                total_time += execution_time

                            # Store detailed results
                            self.results[env_name][planner_name].append({
                                "agent_count": agent_count,
                                "trial": trial,
                                "success": success,
                                "steps": steps if success else None,
                                "tokens": tokens,
                                "api_calls": api_calls,
                                "execution_time": execution_time,
                                "plan": plan
                            })
                        except Exception as e:
                            print(f"Error running {planner_name} on {env_name} with {agent_count} agents: {e}")
                            self.results[env_name][planner_name].append({
                                "agent_count": agent_count,
                                "trial": trial,
                                "success": False,
                                "error": str(e)
                            })

                    # Calculate averages for successful runs
                    success_rate = success_count / num_trials
                    avg_tokens = total_tokens / success_count if success_count > 0 else 0
                    avg_api_calls = total_api_calls / success_count if success_count > 0 else 0
                    avg_steps = total_steps / success_count if success_count > 0 else 0
                    avg_time = total_time / success_count if success_count > 0 else 0

                    # Store summary metrics
                    self.summary[env_name][planner_name][agent_count] = {
                        "success_rate": success_rate,
                        "avg_tokens": avg_tokens,
                        "avg_api_calls": avg_api_calls,
                        "avg_steps": avg_steps,
                        "avg_time": avg_time
                    }

                    print(f"{planner_name} results with {agent_count} agents:")
                    print(f"  Success rate: {success_rate:.2%}")
                    print(f"  Avg tokens: {avg_tokens:.2f}")
                    print(f"  Avg API calls: {avg_api_calls:.2f}")
                    print(f"  Avg steps: {avg_steps:.2f}")
                    print(f"  Avg execution time: {avg_time:.2f}s")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f"{self.output_dir}/detailed_results_{timestamp}.json", "w") as f:
            json.dump(self.results, f, indent=2)

        with open(f"{self.output_dir}/summary_results_{timestamp}.json", "w") as f:
            json.dump(self.summary, f, indent=2)

        # Generate plots
        self.generate_plots(timestamp)

        return self.summary

    def generate_plots(self, timestamp):
        """Generate plots from test results"""
        # Create plots directory
        plots_dir = f"{self.output_dir}/plots_{timestamp}"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Plot success rates
        self._plot_metric("success_rate", "Success Rate", plots_dir)

        # Plot average tokens
        self._plot_metric("avg_tokens", "Average Tokens", plots_dir)

        # Plot average API calls
        self._plot_metric("avg_api_calls", "Average API Calls", plots_dir)

        # Plot average steps
        self._plot_metric("avg_steps", "Average Steps", plots_dir)

    def _plot_metric(self, metric, title, plots_dir):
        """Helper method to plot a specific metric"""
        for env_name in self.summary.keys():
            plt.figure(figsize=(10, 6))

            for planner_name in self.summary[env_name].keys():
                agent_counts = sorted(self.summary[env_name][planner_name].keys())
                values = [self.summary[env_name][planner_name][ac][metric] for ac in agent_counts]

                plt.plot(agent_counts, values, marker='o', label=planner_name)

            plt.title(f"{title} - {env_name}")
            plt.xlabel("Number of Agents")
            plt.ylabel(title)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            plt.savefig(f"{plots_dir}/{env_name}_{metric}.png")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Batch testing framework for multi-robot planning")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per configuration")
    parser.add_argument("--agents", type=int, nargs="+", default=[4, 8, 16, 32], help="Agent counts to test")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()

    tester = BatchTester(output_dir=args.output)
    summary = tester.run_tests(num_trials=args.trials, agent_counts=args.agents)

    print("\n=== Testing Complete ===")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
