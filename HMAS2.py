import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import BoxNet1
import BoxNet2

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class HMAS2:
    def __init__(self, environment_type="boxnet1"):
        self.token_count = 0
        self.environment_type = environment_type
        self.env = BoxNet1.BoxNet1() if environment_type == "boxnet1" else BoxNet2.BoxNet2()
        self.state_action_history = []

    def format_central_prompt(self, feedback=None):
        lines = ["You are a centralized planner. Provide a plan as a JSON object."]
        for i, agent in enumerate(self.env.agents):
            lines.append(f"- Agent {i} at {agent.position}")
        for box in self.env.boxes:
            lines.append(f"- {box.color} box at {box.position if hasattr(box,'position') else box.positions}")
        for color, positions in self.env.goals.items():
            lines.append(f"- Goal for {color} at {positions[0]}")
        if feedback:
            lines.append("\nPrevious agent feedback:")
            for k, v in feedback.items():
                lines.append(f"{k}: {v}")
        lines.append("\nReturn: {\"Agent0\": \"move...\", ...}")
        return "\n".join(lines)

    def format_local_prompt(self, agent_id, action):
        return f"You are Agent {agent_id}. Do you agree with your assigned action: '{action}'?\nRespond ONLY with:\n- AGREE\n- DISAGREE: [reason]"

    def call_llm(self, prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful agent."}, {"role": "user", "content": prompt}],
            temperature=0
        )
        self.token_count += response.usage.total_tokens
        return response.choices[0].message.content.strip()

    def run_planning(self, max_iterations=5):
        print("--- HMAS-2 Planning ---")
        feedback = None

        for iteration in range(max_iterations):
            central_prompt = self.format_central_prompt(feedback)
            response = self.call_llm(central_prompt)

            try:
                current_plan = json.loads(response)
            except json.JSONDecodeError:
                print("Central plan invalid")
                return None

            feedback = {}
            all_agree = True

            for agent_id, action in current_plan.items():
                idx = int(agent_id.replace("Agent", ""))
                prompt = self.format_local_prompt(idx, action)
                agent_response = self.call_llm(prompt)
                feedback[agent_id] = agent_response
                if not agent_response.startswith("AGREE"):
                    all_agree = False

            if all_agree:
                print("✅ All agents agreed.")
                return current_plan

        print("❌ No consensus reached.")
        return None