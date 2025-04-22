import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import BoxNet1
import BoxNet2

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class HMAS1:
    def __init__(self, environment_type="boxnet2"):
        # Initialize either BoxNet1 or BoxNet2
        if environment_type.lower() == "boxnet1":
            self.env = BoxNet1.BoxNet1()
            self.env_type = "boxnet1"
        else:
            self.env = BoxNet2.BoxNet2()
            self.env_type = "boxnet2"

        self.turn_history = []
        self.token_count = 0

    def format_central_prompt(self):
        """Create concise prompt for central planner"""
        if self.env_type == "boxnet1":
            # Format state for BoxNet1
            boxes_info = []
            for box in self.env.boxes:
                for i, pos in enumerate(box.positions):
                    goal = self.env.goals[box.color][i]
                    boxes_info.append(f"{box.color}:{pos}->{goal}")

            agents_info = [f"Agent{i}:{a.position}" for i, a in enumerate(self.env.agents)]

            prompt = f"""CENTRAL PLANNER: Assign one action to each agent to move boxes to goals.
STATE:
- Boxes: {','.join(boxes_info)}
- Agents: {','.join(agents_info)}
RULES:
- Agents fixed in cells, can only move boxes in their cell
- Actions: move_box(color,direction), move_to_goal(color), do_nothing()
FORMAT OUTPUT AS JSON: {{"Agent0":"action(params)",...}}"""

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

            prompt = f"""CENTRAL PLANNER: Create initial plan for BoxNet2.
STATE:
- Boxes: {json.dumps(box_positions)}
- Corners: {json.dumps(corner_occupancy)}
- Goals: {json.dumps(goals)}
- Agents: {','.join(agents)}
RULES:
- Agents fixed in cells
- Boxes move between cells ONLY via corners
- Only ONE box per corner allowed
- Actions: move_box_corner_to_corner, move_box_corner_to_goal, do_nothing
AVAILABLE ACTIONS:
{json.dumps(agent_actions)}
FORMAT OUTPUT AS JSON: {{"Agent0":"action(params)",...}}"""

        return prompt

    def format_local_prompt(self, agent_id, current_plan):
        """Create concise prompt for local agent discussion"""
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
CURRENT PLAN: {json.dumps(current_plan)}
DIALOGUE HISTORY:
{json.dumps(self.turn_history[-3:] if len(self.turn_history) > 3 else self.turn_history)}

Propose your action or approve the plan.
RESPOND WITH:
- PROCEED:[your_action] - To continue discussion
- EXECUTE:{{"Agent0":"action",...}} - To finalize plan"""

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
CURRENT PLAN: {json.dumps(current_plan)}
DIALOGUE HISTORY:
{json.dumps(self.turn_history[-3:] if len(self.turn_history) > 3 else self.turn_history)}

Propose your action or approve the plan.
RESPOND WITH:
- PROCEED:[your_action] - To continue discussion
- EXECUTE:{{"Agent0":"action",...}} - To finalize plan"""

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

    def run_planning(self, max_dialogue_rounds=10):
        """Run the full HMAS-1 planning process"""
        print("--- HMAS-1 Planning ---")

        # 1. Central planner creates initial plan
        central_prompt = self.format_central_prompt()
        print("Querying central planner...")
        central_response = self.call_llm(central_prompt)

        try:
            current_plan = json.loads(central_response)
            print(f"Initial plan: {json.dumps(current_plan)}")
        except json.JSONDecodeError:
            print("Error: Central planner didn't return valid JSON")
            print(f"Response: {central_response}")
            return None

        # 2. Local agents engage in dialogue
        active_agent_ids = list(range(len(self.env.agents)))
        self.turn_history = []

        for round_num in range(max_dialogue_rounds):
            print(f"\n-- Dialogue Round {round_num + 1} --")
            execution_plan = None

            for i, agent_id in enumerate(active_agent_ids):
                local_prompt = self.format_local_prompt(agent_id, current_plan)
                print(f"Agent {agent_id} thinking...")
                response = self.call_llm(local_prompt)

                self.turn_history.append({f"Agent{agent_id}": response})
                print(f"Agent {agent_id}: {response[:100]}...")

                if response.startswith("EXECUTE:"):
                    try:
                        plan_str = response.replace("EXECUTE:", "").strip()
                        execution_plan = json.loads(plan_str)
                        print("Consensus reached!")
                        break
                    except json.JSONDecodeError:
                        print("Warning: Invalid execution plan format")

            if execution_plan:
                break

        if execution_plan:
            print(f"\nFinal plan: {json.dumps(execution_plan)}")
            print(f"Total tokens used: {self.token_count}")
            return execution_plan
        else:
            print("Failed to reach consensus within limit")
            return None
