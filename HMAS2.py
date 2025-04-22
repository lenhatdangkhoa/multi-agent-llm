import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import BoxNet1
import BoxNet2
from HMAS1 import HMAS1

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class HMAS2:
    def __init__(self, environment_type="boxnet2"):
        # Initialize either BoxNet1 or BoxNet2
        if environment_type.lower() == "boxnet1":
            self.env = BoxNet1.BoxNet1()
            self.env_type = "boxnet1"
        else:
            self.env = BoxNet2.BoxNet2()
            self.env_type = "boxnet2"

        self.token_count = 0
        self.state_action_history = []  # Only track state-action pairs, not full dialogue

    def format_central_prompt(self, feedback=None):
        """Create concise prompt for central planner with optional feedback"""
        if self.env_type == "boxnet1":
            # Format state for BoxNet1
            boxes_info = []
            for box in self.env.boxes:
                for i, pos in enumerate(box.positions):
                    goal = self.env.goals[box.color][i]
                    boxes_info.append(f"{box.color}:{pos}->{goal}")

            agents_info = [f"Agent{i}:{a.position}" for i, a in enumerate(self.env.agents)]

            prompt = f"""CENTRAL PLANNER: Create/revise plan for BoxNet1.
STATE:
- Boxes: {','.join(boxes_info)}
- Agents: {','.join(agents_info)}
- History: {json.dumps(self.state_action_history[-3:] if self.state_action_history else [])}
RULES:
- Agents fixed in cells, can only move boxes in their cell
- Actions: move_box(color,direction), move_to_goal(color), do_nothing()"""

            if feedback:
                prompt += f"\nAGENT FEEDBACK:\n{json.dumps(feedback)}"
                prompt += "\nAdjust your plan based on feedback."

            prompt += "\nFORMAT OUTPUT AS JSON: {\"Agent0\":\"action(params)\",...}"

        else:  # BoxNet2
            # Get boxnet2 state in compact form
            box_positions = {b.color: b.position for b in self.env.boxes}
            corner_occupancy = {str(k): (v.color if v else "empty")
                                for k, v in {c.position: c.occupied_by for c in self.env.corners}.items()}
            goals = {color: positions for color, positions in self.env.goals.items()}
            agents = [f"Agent{i}:{a.cell_position}" for i, a in enumerate(self.env.agents)]

            # Get available actions for each agent
            agent_actions = {}
            for i, agent in enumerate(self.env.agents):
                agent_actions[f"Agent{i}"] = agent.get_available_actions(self.env)

            prompt = f"""CENTRAL PLANNER: Create/revise plan for BoxNet2.
STATE:
- Boxes: {json.dumps(box_positions)}
- Corners: {json.dumps(corner_occupancy)}
- Goals: {json.dumps(goals)}
- Agents: {','.join(agents)}
- History: {json.dumps(self.state_action_history[-3:] if self.state_action_history else [])}
RULES:
- Agents fixed in cells
- Boxes move between cells ONLY via corners
- Only ONE box per corner allowed
- Actions: move_box_corner_to_corner, move_box_corner_to_goal, do_nothing
AVAILABLE ACTIONS:
{json.dumps(agent_actions)}"""

            if feedback:
                prompt += f"\nAGENT FEEDBACK:\n{json.dumps(feedback)}"
                prompt += "\nAdjust your plan based on feedback."

            prompt += "\nFORMAT OUTPUT AS JSON: {\"Agent0\":\"action(params)\",...}"

        return prompt

    def format_local_prompt(self, agent_id, assigned_action):
        """Create concise prompt for local agent feedback"""
        agent = self.env.agents[agent_id]

        if self.env_type == "boxnet1":
            cell_pos = agent.position
            # Get boxes and goals in this cell
            cell_boxes = [b for b in self.env.boxes
                          if any(pos == cell_pos for pos in b.positions)]
            cell_goals = {color: positions for color, positions in self.env.goals.items()
                          if cell_pos in positions}

            prompt = f"""LOCAL AGENT{agent_id} at {cell_pos}:
YOUR VIEW:
- Boxes in cell: {[b.color for b in cell_boxes]}
- Goals in cell: {cell_goals}
ASSIGNED ACTION: {assigned_action}

Check if this action is valid and safe.
RESPOND WITH ONLY:
- AGREE - If action is valid and safe
- DISAGREE:[reason] - If action is problematic"""

        else:  # BoxNet2
            cell_pos = agent.cell_position

            # Get local view (corners connected to this cell)
            cell_corners = []
            for corner in self.env.corners:
                if cell_pos in corner.connected_cells:
                    cell_corners.append({
                        "position": corner.position,
                        "occupied_by": corner.occupied_by.color if corner.occupied_by else None
                    })

            # Get available actions for this agent
            available_actions = agent.get_available_actions(self.env)

            prompt = f"""LOCAL AGENT{agent_id} at {cell_pos}:
YOUR VIEW:
- Connected corners: {json.dumps(cell_corners)}
- Available actions: {json.dumps(available_actions)}
ASSIGNED ACTION: {assigned_action}

Check if this action is valid and safe (no collisions).
RESPOND WITH ONLY:
- AGREE - If action is valid and safe
- DISAGREE:[reason] - If action is problematic"""

        return prompt

    def call_llm(self, prompt):
        """Query LLM with prompt and track token usage"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        self.token_count += response.usage.total_tokens
        return response.choices[0].message.content

    def run_planning(self, max_iterations=5):
        """Run the full HMAS-2 planning process"""
        print("--- HMAS-2 Planning ---")

        for iteration in range(max_iterations):
            print(f"\n-- Iteration {iteration + 1} --")

            # 1. Central planner creates/revises plan
            central_prompt = self.format_central_prompt() if iteration == 0 else self.format_central_prompt(feedback)
            print("Querying central planner...")
            central_response = self.call_llm(central_prompt)

            try:
                current_plan = json.loads(central_response)
                print(f"Proposed plan: {json.dumps(current_plan)}")
            except json.JSONDecodeError:
                print("Error: Central planner didn't return valid JSON")
                print(f"Response: {central_response}")
                return None

            # 2. Local agents provide feedback
            feedback = {}
            all_agree = True

            for agent_id, action in current_plan.items():
                agent_idx = int(agent_id.replace("Agent", ""))
                if agent_idx >= len(self.env.agents):
                    feedback[agent_id] = "DISAGREE:Invalid agent ID"
                    all_agree = False
                    continue

                local_prompt = self.format_local_prompt(agent_idx, action)
                print(f"Agent {agent_id} checking...")
                response = self.call_llm(local_prompt).strip()

                feedback[agent_id] = response
                print(f"Agent {agent_id}: {response}")

                if response.startswith("DISAGREE"):
                    all_agree = False

            # 3. Check for consensus
            if all_agree:
                print("Consensus reached!")

                # Add to state-action history before returning
                self.state_action_history.append({
                    "state": "abbreviated_state",
                    "action": current_plan
                })

                print(f"\nFinal plan: {json.dumps(current_plan)}")
                print(f"Total tokens used: {self.token_count}")
                return current_plan

        print("Failed to reach consensus within iteration limit")
        return None