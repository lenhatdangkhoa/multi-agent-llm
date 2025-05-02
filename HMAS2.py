import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import BoxNet1
import BoxNet2_test

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class HMAS2:
    def __init__(self, environment_type="boxnet1"):
        self.token_count = 0
        self.environment_type = environment_type
        self.env = BoxNet1.BoxNet1() if environment_type == "boxnet1" else BoxNet2_test.BoxNet2()
        self.state_action_history = []

    def format_central_prompt(self, feedback=None):
        lines = ["You are a centralized planner. Provide a plan as a JSON object."]

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
                lines.append(f"- Agent {i} at {agent.position}")
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

        # Add feedback if provided
        if feedback:
            lines.append("\nPrevious agent feedback:")
            for k, v in feedback.items():
                lines.append(f"{k}: {v}")

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

        lines.append("\nReturn the plan ONLY as a JSON object like this example:")
        lines.append("{\"Agent0\": \"move blue box from (0,0) to (1,0) down\", \"Agent1\": \"do nothing\"}")

        return "\n".join(lines)

    def format_local_prompt(self, agent_id, action):
        return (
            f"You are Agent {agent_id}. Do you agree with your assigned action: '{action}'?\n"
            "Respond ONLY with:\n- AGREE\n- DISAGREE: [reason]"
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
