import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import BoxNet1
import BoxNet2

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class HMAS1:
    def __init__(self, environment_type="boxnet1"):
        self.token_count = 0
        self.environment_type = environment_type
        self.env = BoxNet1.BoxNet1() if environment_type == "boxnet1" else BoxNet2.BoxNet2()
        self.turn_history = []

    def format_central_prompt(self):
        lines = ["You are a centralized planner. Your task is to propose a multi-agent plan."]
        for i, agent in enumerate(self.env.agents):
            lines.append(f"- Agent {i} at {agent.position}")
        for box in self.env.boxes:
            lines.append(f"- {box.color} box at {box.position if hasattr(box,'position') else box.positions}")
        for color, positions in self.env.goals.items():
            lines.append(f"- Goal for {color} at {positions[0]}")
        lines.append("\nRespond in JSON format as {\"Agent0\": \"move red box from (x,y) to (x,y) right\", ...}")
        return "\n".join(lines)

    def format_local_prompt(self, agent_id, current_plan):
        plan_str = json.dumps(current_plan, indent=2)
        return f"You are Agent {agent_id}. Given this shared plan, do you agree?\n{plan_str}\nRespond with:\n- EXECUTE: {{full_plan}} if agreed\n- PROCEED: explanation if not ready."

    def call_llm(self, prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful agent."}, {"role": "user", "content": prompt}],
            temperature=0
        )
        self.token_count += response.usage.total_tokens
        return response.choices[0].message.content

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
