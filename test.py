import re
import BoxNet1
def parse_llm_plan(text):
    actions = []

    pattern_move = r"- Agent (\d+): move (\w+) box from \((\d+), (\d+)\) to \((\d+), (\d+)\)"
    pattern_nothing = r"- Agent (\d+): do nothing"

    for line in text.strip().split('\n'):
        move_match = re.match(pattern_move, line.strip())
        nothing_match = re.match(pattern_nothing, line.strip())

        if move_match:
            agent_id = int(move_match.group(1))
            color = move_match.group(2)
            from_pos = (int(move_match.group(3)), int(move_match.group(4)))
            to_pos = (int(move_match.group(5)), int(move_match.group(6)))

            direction = get_direction(from_pos, to_pos)
            actions.append((agent_id, color, from_pos, direction))
        
        elif nothing_match:
            agent_id = int(nothing_match.group(1))
            actions.append((agent_id, "none", None, "stay"))

    return actions

def get_direction(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == -1: return "up"
    if dx == 1: return "down"
    if dy == -1: return "left"
    if dy == 1: return "right"
    return "stay"

actions = parse_llm_plan("""- Agent 0: move blue box from (0, 0) to (1, 0) [down]
- Agent 4: move blue box from (1, 0) to (1, 1) [right]
- Agent 1: move yellow box from (0, 1) to (1, 1) [down]
- Agent 5: move yellow box from (1, 1) to (1, 0) [left]
- Agent 3: move yellow box from (0, 3) to (1, 3) [down]
- Agent 6: move red box from (1, 2) to (0, 2) [up]""")

#print(actions)
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
        else:
            print(f"⚠️ Agent {agent_id} could not find {color} box at {from_pos}")

    print("\n🧱 Final Environment State:")
    print(env.goals)
    for box in env.boxes:
        print(f"{box.color} box positions: {box.positions}")

execute_plan(BoxNet1.BoxNet1(), actions)

