from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import BoxNet1
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CELL_GRID = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 2x4 grid flattened
NUM_AGENTS = 8

# DMAS state
turn_history = []

def build_prompt(agent_id, boxes, goals, turn_history):
    # Extract this agent's cell position
    row, col = agent_id // 4, agent_id % 8

    for g in goals.values():
        for val in g:
            print(val)
            if (row, col) == g:
                print(f"Agent {agent_id} cell goal: {g}")
    cell_boxes = [b.positions for b in boxes if (row, col) in b.positions]
    cell_goals = [g for g in goals.values() if (row, col) in g]
    #print(f"Agent {agent_id} cell boxes: {cell_boxes}, cell goals: {cell_goals}")
    prompt = f"""You are Robot R{agent_id}, responsible for cell ({row}, {col}).
                Your job is to suggest an action to help move boxes to their goals.

                Boxes in your cell: {json.dumps(cell_boxes)}
                Goals in your cell: {json.dumps(cell_goals)}

                Previous robots said:
                {json.dumps(turn_history)}

                Respond with one of the following formats:
                1. "ACTION: move box [box_id] to cell (target_row, target_col)"
                2. "ACTION: move box [box_id] to goal"
                3. "ACTION: do nothing"
                4. "EXECUTE" (if ready to execute all plans)
                """
    return prompt

def query_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message["content"]

def dmas_plan(boxes, goals):
    global turn_history
    turn_history = []
    actions = []
    print(goals)
    print(boxes)
    for agent_id in range(NUM_AGENTS):
        prompt = build_prompt(agent_id, boxes, goals, turn_history)
        #reply = query_llm(prompt).strip()
        print(f"R{agent_id}: {reply}")
        turn_history.append({f"R{agent_id}": reply})
        if "EXECUTE" in reply:
            break

    for entry in turn_history:
        for k, v in entry.items():
            if "ACTION:" in v:
                actions.append((k, v.replace("ACTION:", "").strip()))

    return actions  # list of tuples (robot_id, action string)

print(dmas_plan(BoxNet1.BoxNet1().boxes, BoxNet1.BoxNet1().goals))
