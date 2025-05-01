import BoxNet1
import BoxNet2
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(env):
    """Build prompt for DMAS decentralized agent planning."""
    lines = [
        "You are a distributed robot agent operating independently in a grid environment.",
    ]

    if isinstance(env, BoxNet1.BoxNet1):
        lines.append(f"Grid size: {env.GRID_WIDTH} rows x {env.GRID_HEIGHT} columns (BoxNet1)")
        lines.append("\nBoxes:")
        for box in env.boxes:
            for pos in box.positions:
                lines.append(f"- {box.color} box at {pos}")
    elif isinstance(env, BoxNet2.BoxNet2):
        lines.append(f"Environment: BoxNet2 using corner-based movement.")
        lines.append("\nBoxes:")
        for box in env.boxes:
            lines.append(f"- {box.color} box at corner {box.position}")

    lines.append("\nAgents:")
    for i, agent in enumerate(env.agents):
        lines.append(f"- Agent {i} at {agent.position}")

    lines.append("\nEach agent must independently decide on an action.")
    lines.append("Format your responses as: '- Agent [id]: move [color] box from (x, y) to (x, y) [direction]' or '- Agent [id]: do nothing'")
    return "\n".join(lines)

def query_llm(prompt):
    """Send prompt to OpenAI and get response."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a decentralized multi-agent system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content, response.usage.total_tokens

def apply_action(env, agent_id, box_color, from_pos, direction):
    """Move a box based on agent's action."""
    if isinstance(env, BoxNet1.BoxNet1):
        box = next((b for b in env.boxes if b.color == box_color and from_pos in b.positions), None)
        if box:
            return env.move_box(box, from_pos, direction)
    elif isinstance(env, BoxNet2.BoxNet2):
        box = next((b for b in env.boxes if b.color == box_color and b.position == from_pos), None)
        if box:
            return env.move_box(box, from_pos, direction)
    return False

def dmas_plan(boxes, goals):
    global turn_history
    turn_history = []
    actions = []
    
    while True:
        for agent_id in range(NUM_AGENTS):
            prompt = build_prompt(agent_id, boxes, goals, turn_history)
            reply = query_llm(prompt).strip()
            print(f"R{agent_id}: {reply}")
            turn_history.append({f"R{agent_id}": reply})
            if "ACTION" in reply:
                apply_action(reply, boxes)
            elif "EXECUTE" in reply:
                print("Execution phase triggered.")
        break
    return actions  # or do execution here if you want

def dmas_plan(plan_text, env):
    """Parse and apply DMAS plan to the environment."""
    pattern_move = r"- Agent (\\d+): move (\\w+) box from \\((\\d+), (\\d+)\\) to \\((\\d+), (\\d+)\\) (\\w+)"
    pattern_nothing = r"- Agent (\\d+): do nothing"

    for line in plan_text.strip().split('\\n'):
        move_match = re.match(pattern_move, line.strip())
        nothing_match = re.match(pattern_nothing, line.strip())

        if move_match:
            agent_id = int(move_match.group(1))
            color = move_match.group(2)
            from_pos = (int(move_match.group(3)), int(move_match.group(4)))
            direction = move_match.group(7)
            apply_action(env, agent_id, color, from_pos, direction)
        elif nothing_match:
            agent_id = int(nothing_match.group(1))
            print(f"Agent {agent_id} does nothing.")

def run_dmas(env):
    """Wrapper to run DMAS planning."""
    prompt = build_prompt(env)
    plan_text, total_tokens = query_llm(prompt)
    return plan_text, total_tokens
