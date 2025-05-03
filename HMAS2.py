import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import BoxNet1
from HMAS1 import HMAS1
import BoxNet2_test
import time
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class HMAS2:
    def __init__(self, environment_type="boxnet1"):
        self.token_count = 0
        self.environment_type = environment_type
        self.env = BoxNet1.BoxNet1() if environment_type == "boxnet1" else BoxNet2_test.BoxNet2()

    def format_central_prompt(self):
        # Use the same logic from your HMAS1 to format the initial plan prompt
        hmas1 = HMAS1(self.environment_type)
        return hmas1.format_central_prompt(hmas1.env)

    def format_feedback_prompt(self, agent_id, agent, action):
        lines = [
            f"You are Agent {agent_id}. Your job is to review your assigned action:",
            f"- Assigned Action: {action}",
            "You must respond with either:",
            "- 'agree' if the action is acceptable, or",
            "- 'suggest: [your better action]' if you want to suggest an alternative."
        ]
        return "\n".join(lines)

    def call_llm(self, prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        self.token_count += response.usage.total_tokens
        return response.choices[0].message.content.strip(), self.token_count

    def parse_llm_plan(self, text):
        actions = []
        pattern_move = r".*?Agent (\d+): move (\w+) box from \((\d+), (\d+)\) to \((\d+), (\d+)\)(?: \[?(\w+)]?)?"
        pattern_nothing = r".*?Agent (\d+): do nothing"
        pattern_move_to_goal = r".*?Agent (\d+): move (\w+) box to goal"

        for line in text.strip().split('\n'):
            move_match = re.match(pattern_move, line.strip())
            nothing_match = re.match(pattern_nothing, line.strip())
            move_to_goal_match = re.match(pattern_move_to_goal, line.strip())

            if move_match:
                agent_id = int(move_match.group(1))
                color = move_match.group(2)
                from_pos = (int(move_match.group(3)), int(move_match.group(4)))
                direction = move_match.group(7)
                actions.append((agent_id, color, from_pos, direction))
            elif nothing_match:
                agent_id = int(nothing_match.group(1))
                actions.append((agent_id, "none", None, "stay"))
            elif move_to_goal_match:
                agent_id = int(move_to_goal_match.group(1))
                color = move_to_goal_match.group(2)
                actions.append((agent_id, color, None, "goal"))
        return actions

    def execute_plan(self, env, actions):
        for agent_id, color, from_pos, direction in actions:
            if color == "none":
                print(f"Agent {agent_id} does nothing")
                continue
            if direction == "goal":
                env.move_to_goal(color)
                continue
            box = next((b for b in env.boxes if b.color == color and from_pos in b.positions), None)
            if box:
                success = env.move_box(box, from_pos, direction)
                print(f"{'✅ Success' if success else '❌ Failed'}: Agent {agent_id} moved {color} box from {from_pos} {direction}")
            else:
                print(f"⚠️ Agent {agent_id} could not find {color} box at {from_pos}")
            time.sleep(0.5)
        print("\n🧱 Final Environment State:")
        print(self.env.goals)
        for box in env.boxes:
            print(f"{box.color} box positions: {box.positions}")

    def runHMAS2(self):
        print("\n== Central Planner Proposing Initial Plan ==")
        central_prompt = self.format_central_prompt()
        central_plan, _ = self.call_llm(central_prompt)
        print(central_plan)

        consensus_reached = False
        for round_num in range(3):
            print(f"\n== Feedback Round {round_num+1} ==")
            agent_feedback = []

            for id, agent in enumerate(self.env.agents):
                # Get just this agent's action line
                pattern = rf"Agent {id}:.*"
                match = re.search(pattern, central_plan)
                action_line = match.group(0) if match else "do nothing"
                
                prompt = self.format_feedback_prompt(id, agent, action_line)
                feedback, _ = self.call_llm(prompt)
                print(f"Agent {id} Feedback: {feedback}")
                agent_feedback.append((id, feedback.strip()))

            if all(fb == "agree" for _, fb in agent_feedback):
                consensus_reached = True
                break
            else:
                feedback_summary = "\n".join([f"Agent {id}: {fb}" for id, fb in agent_feedback])
                central_prompt += f"\n\nAgents provided feedback on the plan:\n{feedback_summary}\nPlease revise the plan."
                central_plan, _ = self.call_llm(central_prompt)
                print("\n🔁 Revised Plan:\n", central_plan)

        final_actions = self.parse_llm_plan(central_plan)
        self.execute_plan(self.env, final_actions)

if __name__ == "__main__":
    hmas2 = HMAS2(environment_type="boxnet2")
    hmas2.runHMAS2()
