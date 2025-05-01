import re
import time
import BoxNet2_test
temp = """
- Agent 1: move blue box from (0, 2) to (0, 1) left
- Agent 0: move blue box from (0, 1) to (0, 0) left
- Agent 0: move blue box to goal
- Agent 3: move yellow box from (2, 2) to (2, 1) left
- Agent 2: move yellow box from (2, 1) to (2, 0) down
- Agent 2: move yellow box to goal
- Agent 3: move green box from (2, 1) to (1, 1) up
- Agent 1: move green box from (1, 1) to (0, 1) up
- Agent 1: move green box to goal
- Agent 0: do nothing
- Agent 2: do nothing
- Agent 3: do nothing
"""

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
            print("here")
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
def execute_plan(env, actions):
    for agent_id, color, from_pos, direction in actions:
        if color == "none":
            print(f"Agent {agent_id} does nothing")
            continue
        if direction == "goal":
            env.move_to_goal(color)
            continue
        # Find the box object by color and current position
        box = next(
            (b for b in env.boxes if b.color == color and from_pos in b.positions),
            None
        )

        if box:
            success = env.move_box(box, from_pos, direction)
            status = "✅ Success" if success else "❌ Failed"
            #print(f"{status}: Agent {agent_id} moved {color} box from {from_pos} {direction}")
            if not success:
                return False
        else:
            print(f"⚠️ Agent {agent_id} could not find {color} box at {from_pos}")
            return False
        
        time.sleep(1) # For debugging
    print("\n🧱 Final Environment State:")
    print(env.goals)
    for box in env.boxes:
        print(f"{box.color} box positions: {box.positions}")
    
    
    return True
env = BoxNet2_test.BoxNet2()
#print(parse_llm_plan(temp))
actions = parse_llm_plan(temp)
execute_plan(env, actions)
