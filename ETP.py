from openai import OpenAI
import json
import re
import os
from dotenv import load_dotenv
import BoxNet1
import time


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def intialPlan(env: BoxNet1):
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
    return "\n".join(lines)

def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "system", "content": "You are a helpful robot task planner."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    total_tokens = response.usage.total_tokens
    print(f"Total tokens used: {total_tokens}")
    return response.choices[0].message.content

def parse_llm_plan(text):
    actions = []

    pattern_move = r"- Agent (\d+): move (\w+) box from \((\d+), (\d+)\) to \((\d+), (\d+)\) (\w+)"
    pattern_nothing = r"- Agent (\d+): do nothing"

    for line in text.strip().split('\n'):
        move_match = re.match(pattern_move, line.strip())
        nothing_match = re.match(pattern_nothing, line.strip())

        if move_match:
            agent_id = int(move_match.group(1))
            color = move_match.group(2)
            from_pos = (int(move_match.group(3)), int(move_match.group(4)))
            to_pos = (int(move_match.group(5)), int(move_match.group(6)))

            direction = move_match.group(7)
            actions.append((agent_id, color, from_pos, direction))
        
        elif nothing_match:
            agent_id = int(nothing_match.group(1))
            actions.append((agent_id, "none", None, "stay"))

    return actions

def execute_plan(env, actions):
    for agent_id, color, from_pos, direction in actions:
        if color == "none":
            print(f"Agent {agent_id} does nothing")
            continue

        # Find the box object by color and current position
        box = next(
            (b for b in env.boxes if b.color == color and from_pos in b.positions),
            None
        )

        if box:
            success = env.move_box(box, from_pos, direction)
            status = "✅ Success" if success else "❌ Failed"
            print(f"{status}: Agent {agent_id} moved {color} box from {from_pos} {direction}")
            if not success:
                return False
        else:
            print(f"⚠️ Agent {agent_id} could not find {color} box at {from_pos}")
            return False
        
        time.sleep(1)
    print("\n🧱 Final Environment State:")
    print(env.goals)
    for box in env.boxes:
        print(f"{box.color} box positions: {box.positions}")
    
    return True

def runETP():
    env = BoxNet1.BoxNet1()
    prompt = intialPlan(env)
    response = call_llm(prompt)
    actions = parse_llm_plan(response)
    while (not execute_plan(env, actions)):
        print(env.goals)
        for box in env.boxes:
            print(f"{box.color} box positions: {box.positions}")
        prompt = intialPlan(env)
        response = call_llm(prompt)
        actions = parse_llm_plan(response)
    
if __name__ == "__main__":
    runETP()