import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import BoxNet1
import BoxNet2_test
import time
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class HMAS1:
    def __init__(self, environment_type="boxnet1"):
        self.token_count = 0
        self.environment_type = environment_type
        self.env = BoxNet1.BoxNet1() if environment_type == "boxnet1" else BoxNet2_test.BoxNet2()
        self.turn_history = []
    def format_central_prompt(self, env):
        """Format prompt for centralized CMAS planner."""
        if isinstance(env, BoxNet1.BoxNet1):
            lines = [
            "You are a centralized task planner for a grid-based environment.",
            "The grid is represented as a 2D array, where each cell can contain a box or an agent.",
            "Your job is to assign each agent to move a colored box to its goal.",
            "If the colored box has multiple goals, the agent should move it to any goal. Make sure all the colored boxes are moved to their goals.",
            "Each agent can only move boxes that are within its own grid cell.",
            "Each agent is stuck in its own cell and can only move boxes to adjacent cells.",
            "A goal cannot be occupied by more than one box.",
            "Multiple boxes can occupy the same cell.",
            "There are three possible actions for each agent: move box to an ajacent cell (up, down, left, right),"
            "move box to the goal cell if the box is in the same cell, or do nothing.",
            f"Grid size: {env.GRID_WIDTH} rows x {env.GRID_HEIGHT} columns\n",
            "Boxes:"
        ]
            for box in env.boxes:
                for i, pos in enumerate(box.positions):
                    goal = env.goals[box.color][i]
                    lines.append(f"- {box.color} box at {pos}, goal at {goal}")

            lines.append("\nAgents:")
            for i, agent in enumerate(env.agents):
                lines.append(f"- Agent {i} at {agent.position}")

            lines.append("\nFor example, if the blue box is at (0, 0) and the goal is at (1, 1),")
            lines.append("the agent can move the blue box from (0, 0) to (1, 0) and then to (1, 1).")
            lines.append("\nAnother example, if the blue goal is at (0, 0), then only one blue box can be moved to (0, 0).")
            lines.append("If there are two blue boxes, then the agent has to find another goal for one of the boxes.")
            lines.append("\nPlease return an ordered list of actions in the following format:")
            lines.append("- Agent [id]: move [color] box from (x, y) to (new x, new y) [direction]")

        elif isinstance(env, BoxNet2_test.BoxNet2):
            lines = [
            "You are a centralized task planner for a grid-based environment.",
            "The grid is represented as a 2D array, where each cell can contain a box",
            "Your job is to assign each agent to move a colored box to its goal.",
            "EAn agent should move each box to its valid goals.",        
            "Each agent can only move boxes that are around its corners",
            "For example, agent 1 is responsible for the corners (0,0), (0,1), (1,0), and (1,1).",
            "Boxes can only be moved at the corners (indicated by their coordinates).",
            "There are three possible actions for each agent: (1) move a box from one corner to another, (2) move a box from a corner to a goal location within the same cell, or (3) do nothing.",    
            "Directions are: up (x-1), down (x+1), left (y-1), right (y+1).",    
            f"Grid size: {env.GRID_WIDTH} rows x {env.GRID_HEIGHT} columns\n",
            "Boxes:"
        ]
            for box in env.boxes:
                for i, pos in enumerate(box.positions):
                    goal = env.goals[box.color]
                    lines.append(f"- {box.color} box at {pos}, goal at {goal}")

            lines.append("\nAgents:")
            for i, agent in enumerate(env.agents):
                lines.append(f"- Agent {i} responsible for cells {agent.position}")

            lines.append("\nEach agent is responsible for four corners of its own cell.")
            lines.append("\nIf the blue box is at (0, 1) and the goal is at (0,0), (0,1), (1,0), (1,1), then agent 1 can move this blue box to goal.")
            lines.append("\nYou don't need to say your thought process, just say the action.")
            lines.append("\nPlease return an ordered list of actions in one of the following formats:")
            lines.append("- Agent [id]: move [color] box from (x, y) to (new x, new y) [direction]")
            lines.append("- Agent [id]: do nothing")
            lines.append("- Agent [id]: move [color] box to goal")
        return "\n".join(lines)
    # def format_central_prompt(self):
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
    def format_local_prompt(self, agent_id, agent, central_plan):
        lines = [
        "You are a local LLM agent responsible for checking and revising your assigned action.",
        f"You are Agent {agent_id}. Your current position or cell is {agent.position}.",
        "Here is the proposed plan from the central planner:"
    ]
        lines.append(central_plan)
        lines.append("\nPlease respond with only your own action in the format:")
        lines.append("- Agent [id]: move [color] box from (x, y) to (new x, new y) [direction] or do nothing or move box to goal.")
        return "\n".join(lines)

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
        return response.choices[0].message.content, self.token_count

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
            time.sleep(1)
        print("\n🧱 Final Environment State:")
        print(self.env.goals)
        for box in self.env.boxes:
            print(f"{box.color} box positions: {box.positions}")

    def runHMAS1(self):
        api_calls = 1
        print("\n== Central Planner Proposing Initial Plan ==")
        initial_prompt = self.format_central_prompt(self.env)
        central_plan, _ = self.call_llm(initial_prompt)
        print(central_plan)

        print("\n== Local Agents Checking and Revising Plan ==")
        local_action_strs = []
        for id, agent in enumerate(self.env.agents):
            agent_prompt = self.format_local_prompt(id, agent, central_plan)
            response, tokens = self.call_llm(agent_prompt)
            api_calls += 1
            print(f"Agent {id} Response:\n{response}\n")
            local_action_strs.append(response)

        final_plan = "\n".join(local_action_strs)
        actions = self.parse_llm_plan(final_plan)
        print("\n== Final Plan ==")
        print(final_plan)
        #self.execute_plan(self.env, actions)

        return final_plan, api_calls

# hmas = HMAS1(environment_type="boxnet2")
# hmas.runHMAS1()