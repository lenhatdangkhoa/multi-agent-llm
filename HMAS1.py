import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import BoxNet1
import BoxNet2_test

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class HMAS1:
    def __init__(self, environment_type="boxnet1"):
        self.token_count = 0
        self.environment_type = environment_type
        self.env = BoxNet1.BoxNet1() if environment_type == "boxnet1" else BoxNet2_test.BoxNet2()
        self.turn_history = []

    def format_central_prompt(self):
        lines = ["You are a centralized planner. Your task is to propose a multi-agent plan."]

        if self.environment_type == "boxnet1":
            lines.append(f"Environment: BoxNet1 with grid size {self.env.GRID_WIDTH}x{self.env.GRID_HEIGHT}")
            lines.append("Each agent is fixed in its cell and can only move boxes to adjacent cells.")
            lines.append("Multiple boxes can occupy the same cell, but a goal can only have one box.")
        else:  # boxnet2
            lines.append(f"Environment: BoxNet2 with grid size {self.env.GRID_WIDTH}x{self.env.GRID_HEIGHT}")
            lines.append("Each agent is responsible for the four corners of its cell.")
            lines.append("Boxes can only be moved between corners, and each corner can only hold one box.")
            lines.append("Avoid placing multiple boxes on the same corner to prevent collisions.")

        # Add agent information
        for i, agent in enumerate(self.env.agents):
            if self.environment_type == "boxnet1":
                lines.append(f"- Agent {i} at position {agent.position}")
            else:
                lines.append(f"- Agent {i} responsible for corners {agent.cell}")

        # Add box information
        for box in self.env.boxes:
            if self.environment_type == "boxnet1":
                lines.append(f"- {box.color} box at positions {box.positions}")
            else:
                for pos in box.positions:
                    lines.append(f"- {box.color} box at corner {pos}")

        # Add goal information
        for color, positions in self.env.goals.items():
            lines.append(f"- Goal for {color} at {positions}")

        # Add action instructions
        lines.append("\nYour plan must only use the following actions for each agent:")
        if self.environment_type == "boxnet1":
            lines.append(
                "- move [color] box from (row,col) to (row,col) [direction] (direction: up, down, left, right)")
            lines.append("- do nothing")
        else:
            lines.append(
                "- move [color] box from (row,col) to (row,col) [direction] (direction: up, down, left, right)")
            lines.append("- move [color] box to goal")
            lines.append("- do nothing")

        lines.append("\nRespond ONLY in JSON format like this example:")
        lines.append("{\"Agent0\": \"move red box from (1,2) to (0,2) up\", \"Agent1\": \"do nothing\"}")

        return "\n".join(lines)

    def format_local_prompt(self, agent_id, current_plan):
        plan_str = json.dumps(current_plan, indent=2)
        return (
            f"You are Agent {agent_id}. Given this shared plan, do you agree?\n{plan_str}\n"
            "Respond with:\n- EXECUTE: {full_plan} if agreed\n- PROCEED: explanation if not ready."
        )

    def call_llm(self, prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        self.token_count += response.usage.total_tokens
        return response.choices[0].message.content.strip()

    def run_planning(self, max_dialogue_rounds=10):
        print("--- HMAS-1 Planning ---")
        central_prompt = self.format_central_prompt()
        central_response = self.call_llm(central_prompt)
        try:
            current_plan = json.loads(central_response)
        except json.JSONDecodeError:
            print("Central plan invalid")
            return None
        active_agent_ids = list(range(len(self.env.agents)))
        for round_num in range(max_dialogue_rounds):
            print(f"-- Dialogue Round {round_num + 1} --")
            round_responses = []
            execution_plans = []
            for agent_id in active_agent_ids:
                local_prompt = self.format_local_prompt(agent_id, current_plan)
                response = self.call_llm(local_prompt)
                self.turn_history.append({f"Agent{agent_id}": response})
                if response.startswith("EXECUTE:"):
                    try:
                        plan_str = response.replace("EXECUTE:", "").strip()
                        plan = json.loads(plan_str)
                        execution_plans.append(plan)
                    except json.JSONDecodeError:
                        print(f"Invalid EXECUTE format from Agent {agent_id}")
            if len(execution_plans) == len(active_agent_ids):
                if all(p == execution_plans[0] for p in execution_plans):
                    print("✅ All agents reached consensus!")
                    return execution_plans[0]
                else:
                    print("❌ EXECUTE plans do not match")
        print("❌ Failed to reach consensus")
        return None
