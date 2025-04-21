from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import BoxNet1
import re
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CELL_GRID = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 2x4 grid flattened
NUM_AGENTS = 8

# DMAS state
turn_history = []
env = BoxNet1.BoxNet1()
def build_prompt(agent_id, boxes, goals, turn_history):
    # Extract this agent's cell position
    row, col = agent_id // 4, agent_id % 4
    cell_boxes = []
    cell_goals = []
    for c, g in goals.items():
        for val in g:
            if (row, col) == val:
                cell_goals.append({c: (row,col)})
    for b in boxes:
        for val in b.positions:
            if (row, col) == val:
                cell_boxes.append({b.color: (row,col)})
    # cell_boxes = [b.positions for b in boxes if (row, col) == b.positions]
    # cell_goals = [g for g in goals.values() if (row, col) == g]
    #print(f"Agent {agent_id} cell boxes: {cell_boxes}, cell goals: {cell_goals}")
    prompt = f"""You are Robot R{agent_id}, responsible for cell ({row}, {col}).
                Your job is to suggest an action to help move boxes to their goals.
                The boxes must be moved to their goals with the correct color.
                You can move boxes to their goals or to other cells. You can also do nothing.
                If the box is in your cell and there is a goal with the same color as the box, you can move it to the goal.
                You can only talk to adjacent robots, not the whole team.
                The grid is divided into 2 rows and 4 columns, so (0,0) is top left and (1,3) is bottom right.
                Boxes in your cell: {json.dumps(cell_boxes)}
                Goals in your cell: {json.dumps(cell_goals)}

                Previous robots said:
                {json.dumps(turn_history)}

                Respond with one of the following formats:
                1. "ACTION: move [box color] box from (box location) to cell (target_row, target_col)"
                2. "ACTION: move [box color] box (box location) to goal"
                3. "ACTION: do nothing"
                4. "EXECUTE" (if ready to execute all plans)
                """
    return prompt

def query_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    total_tokens = response.usage.total_tokens
    print(f"Total tokens used: {total_tokens}")
    return response.choices[0].message.content
def apply_action(reply, boxes):

    parts = reply.split()
    color = parts[2].strip("[]")
    # if "to goal" in reply:
    #     for box in boxes:
    #         if box.color == color:
    #             # assume box has .positions as a list, put it at goal
    #             box.positions = [BoxNet1.find_goal_position(color)]
    #             break

    match = re.search(r"\((\d+),\s*(\d+)\)\sto cell\s*\((\d+),\s*(\d+)\)", reply)
    if match:
        row = int(match.group(3))
        col = int(match.group(4))
        for box in boxes:
            if box.color == color:
                for pos in range(len(box.positions)):
                    if box.positions[pos] == (int(match.group(1)), int(match.group(2))):
                        box.positions[pos] = [(row, col)]
                        break
    for box in boxes:
        print(f"Box {box.color} is now at {box.positions}")

def dmas_plan(boxes, goals):
    global turn_history
    turn_history = []
    actions = []
    
    while True:
        for agent_id in range(NUM_AGENTS):
            prompt = build_prompt(agent_id, boxes, goals, turn_history)
            #print(prompt)
            reply = query_llm(prompt).strip()
            print(f"R{agent_id}: {reply}")
            turn_history.append({f"R{agent_id}": reply})
            if "ACTION" in reply:
                apply_action(reply, boxes)
            elif "EXECUTE" in reply:
                print("Execution phase triggered.")
        break
    return actions  # or do execution here if you want


    for entry in turn_history:
        for k, v in entry.items():
            if "ACTION:" in v:
                actions.append((k, v.replace("ACTION:", "").strip()))

    return actions  # list of tuples (robot_id, action string)

print(dmas_plan(env.boxes, env.goals))
for box in env.boxes:
    print(f"Box {box.color} is now at {box.positions}")
for goal in env.goals:
    print(f"Goal {goal} is at {env.goals[goal]}")