from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import BoxNet1
import BoxNet2_test
import re
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CELL_GRID = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 2x4 grid flattened
NUM_AGENTS = 8

_MOVE_PAT  = re.compile(r".*?Agent\s*(\d+):\s*move\s+(\w+)\s+box\s+from\s*\((\d+),\s*(\d+)\)\s*to\s*\((\d+),\s*(\d+)\).*?(\bup|\bdown|\bleft|\bright)?", re.I)
_GOAL_PAT  = re.compile(r".*?Agent\s*(\d+):\s*move\s+(\w+)\s+box\s+to\s+goal", re.I)
_NONE_PAT  = re.compile(r".*?Agent\s*(\d+):\s*do\s+nothing", re.I)


# DMAS state
turn_history = []
#env = BoxNet1.BoxNet1()
def build_prompt(env, agent_id, boxes, goals, turn_history):
    # Extract this agent's cell position
    cell_boxes = []
    cell_goals = []
    for c, g in goals.items():
        if g == env.agents[agent_id].position:
            cell_goals.append({c: g})
            
    for b in boxes:
        for val in b.positions:
            if val in env.agents[agent_id].position:
                cell_boxes.append({b.color: val})


    #print(f"Agent {agent_id} cell boxes: {cell_boxes}")
    
    if (isinstance(env, BoxNet1.BoxNet1)):
        row, col = agent_id // 4, agent_id % 4
        cell_boxes = []
        cell_goals = []
        for c, g in goals.items():
            #print(f"Goal {c} is at {g}")
            for val in g:
                if (row, col) == val:
                    cell_goals.append({c: (row,col)})
        for b in boxes:
            for val in b.positions:
                if (row, col) == val:
                    cell_boxes.append({b.color: (row,col)})
        prompt = f"""You are Agent {agent_id}, responsible for cell ({row}, {col}).
                    Your job is to suggest an action to help move boxes to their goals.
                    The boxes must be moved to their goals with the correct color.
                    You can move boxes to their goals or to other cells. You can also do nothing.
                    If the box is in your cell and there is a goal with the same color as the box, you can move it to the goal.
                    You can only talk to adjacent robots, not the whole team.
                    The grid is divided into 2 rows and 4 columns, so (0,0) is top left and (1,3) is bottom right.
                    Boxes in your cell: {json.dumps(cell_boxes)}
                    Goals in your cell: {json.dumps(cell_goals)}

                    Previous robots said:
                    {(turn_history)}

                    Respond with one of the following formats:
                    - Agent [id]: move [color] box from (x, y) to (new x, new y) [direction]
                    - Agent [id]: do nothing
                    - Agent [id]: move [color] box to goal
                    """
    elif (isinstance(env, BoxNet2_test.BoxNet2)):
        prompt = [
        f"You are Agent {agent_id}, responsible for cell ({env.agents[agent_id].position}).",
        "The grid is represented as a 2D array, where each cell can contain a box",
        "Your job is to suggest an action to help move boxes to their goals."
        "Each agent can only move boxes that are around its corners",
        "For example, agent 1 is responsible for the corners (0,0), (0,1), (1,0), and (1,1).",
        "Boxes can only be moved at the corners (indicated by their coordinates).",
        "There are three possible actions for each agent: (1) move a box from one corner to another, (2) move a box from a corner to a goal location within the same cell, or (3) do nothing.",    
        "Directions are: up (x-1), down (x+1), left (y-1), right (y+1).",    
        f"Grid size: {env.GRID_WIDTH} rows x {env.GRID_HEIGHT} columns\n",
        ]

        prompt.append(f"\nBoxes in your cell: {json.dumps(cell_boxes)} Goals in your cell: {json.dumps(cell_goals)}")
    
        prompt.append(f"\nPrevious robots said:{(turn_history)}")
        prompt.append("\nEach agent is responsible for four corners of its own cell.")
        prompt.append("\nIf the blue box is at (0, 1) and the goal is at (0,0), (0,1), (1,0), (1,1), then agent 1 can move this blue box to goal.")
        prompt.append("\nYou don't need to say your thought process, just say the action.")
        prompt.append("\nPlease return an ordered list of actions in one of the following formats:")
        prompt.append("- Agent [id]: move [color] box from (x, y) to (new x, new y) [direction]")
        prompt.append("- Agent [id]: do nothing")
        prompt.append("- Agent [id]: move [color] box to goal")
        prompt = "\n".join(prompt)
    return prompt
def parse_llm_plan(text):
    actions = []

    pattern_move = r".*?Agent (\d+): move (\w+) box from \((\d+), (\d+)\) to \((\d+), (\d+)\)(?: \[?(\w+)\]?)?"
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
            to_pos = (int(move_match.group(5)), int(move_match.group(6)))

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
def query_llm(prompt):
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    toks = resp.usage.total_tokens
    print(f"Total tokens used: {toks}")
    return resp.choices[0].message.content, toks
def apply_action(reply, boxes):

    parts = reply.split()
    color = parts[2].strip("[]")
    match = re.search(r"move (\w+) box from \((\d+),\s*(\d+)\) to cell \((\d+),\s*(\d+)\)", reply)
    if match:
        row = int(match.group(4))
        col = int(match.group(5))
        #print(f"Moving {color} box to cell ({row}, {col})")
        for box in boxes:
            if box.color == color:
                for pos in range(len(box.positions)):
                    if box.positions[pos] == (int(match.group(2)), int(match.group(3))):
                        box.positions[pos] = [(row, col)]
                        break
    for box in boxes:
        print(f"Box {box.color} is now at {box.positions}")

def parse_action(text: str):
    text = text.strip()

    # move …
    if m := _MOVE_PAT.match(text):
        aid, col, r1, c1, _, _, dir_raw = m.groups()
        direction = (dir_raw or "stay").lower()
        return int(aid), col.lower(), (int(r1), int(c1)), direction

    # move to goal …
    if m := _GOAL_PAT.match(text):
        aid, col = m.groups()
        return int(aid), col.lower(), None, "goal"

    # do nothing …
    if m := _NONE_PAT.match(text):
        return int(m.group(1)), "none", None, "stay"

    # fallback
    return -1, "none", None, "stay"

def dmas_plan(env, boxes, goals):
    global turn_history, tokens_used
    turn_history = []

    actions   = []
    api_calls = 0

    # three rounds of dialogue
    for _ in range(3):
        for aid in range(NUM_AGENTS):
            prompt = build_prompt(env, aid, boxes, goals, turn_history)
            print(prompt)
            reply, tokens_used = query_llm(prompt)
            reply = reply.strip()
            act_tuple = parse_action(reply)
            actions.append(act_tuple)
            api_calls += 1
            turn_history.append(reply)

    # return either 2‑ or 3‑tuple
    return actions, api_calls, tokens_used  # tokens are logged in batch tester
# print(dmas_plan(env.boxes, env.goals))
# for box in env.boxes:
#     print(f"Box {box.color} is now at {box.positions}")
# for goal in env.goals:
#     print(f"Goal {goal} is at {env.goals[goal]}")