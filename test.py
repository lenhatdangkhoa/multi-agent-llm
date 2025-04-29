# === Final test.py with corrected parse_llm_plan, simulation, and unified render_environment ===
import pygame
import time
import os
import re
from collections import defaultdict
from BoxNet1 import BoxNet1
from BoxNet2 import BoxNet2
from planner_interface import PLANNERS

# --- PLAN PARSING ---
def parse_llm_plan(text):
    actions = []

    if isinstance(text, dict):
        lines = [f"Agent {k.replace('Agent','')}: {v}" for k, v in text.items()]
    else:
        lines = text.strip().split('\n')
        lines = [line for line in lines if line.strip() and line.strip().lower() != "plan:"]

    dir_map = {"north": "up", "south": "down", "east": "right", "west": "left"}

    pattern_move = r"-? ?Agent (\d+): move (\w+) box from \((\d+), (\d+)\) to \((\d+), (\d+)\) (\w+)"
    pattern_nothing = r"-? ?Agent (\d+): do nothing"
    pattern_stay = r"-? ?Agent (\d+): stay at \((\d+), (\d+)\)"
    pattern_pickup = r"-? ?Agent (\d+): Pick up (\w+) box at \((\d+), (\d+)\) and move to \((\d+), (\d+)\)"
    pattern_standby = r"-? ?Agent (\d+): Standby"

    for line in lines:
        line = line.strip()
        move_match = re.match(pattern_move, line)
        pickup_match = re.match(pattern_pickup, line, re.IGNORECASE)
        nothing_match = re.match(pattern_nothing, line)
        stay_match = re.match(pattern_stay, line)
        standby_match = re.match(pattern_standby, line, re.IGNORECASE)

        if move_match:
            agent_id = int(move_match.group(1))
            color = move_match.group(2)
            from_pos = (int(move_match.group(3)), int(move_match.group(4)))
            raw_dir = move_match.group(7).lower()
            direction = dir_map.get(raw_dir, raw_dir)
            actions.append((agent_id, color, from_pos, direction))
        elif pickup_match:
            agent_id = int(pickup_match.group(1))
            color = pickup_match.group(2).lower()
            from_pos = (int(pickup_match.group(3)), int(pickup_match.group(4)))
            to_pos = (int(pickup_match.group(5)), int(pickup_match.group(6)))
            dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
            if abs(dx) + abs(dy) == 1:
                if dx == -1: direction = "up"
                elif dx == 1: direction = "down"
                elif dy == 1: direction = "right"
                elif dy == -1: direction = "left"
                actions.append((agent_id, color, from_pos, direction))
            else:
                print(f"⚠️ Non-adjacent pickup move ignored: {line}")
        elif nothing_match or standby_match:
            agent_id = int((nothing_match or standby_match).group(1))
            actions.append((agent_id, "none", None, "stay"))
        elif stay_match:
            agent_id = int(stay_match.group(1))
            actions.append((agent_id, "none", None, "stay"))
        else:
            print(f"⚠️ Unrecognized plan line skipped: {line}")

    return actions

# --- ENVIRONMENT RENDERING ---
def render_environment(screen, env):
    COLORS = {
        "blue": (0, 102, 204), "yellow": (255, 204, 0), "red": (204, 0, 0),
        "green": (0, 153, 0), "purple": (153, 0, 153), "goal": (200, 200, 200),
        "agent": (50, 205, 50), "background": (255, 255, 255), "grid": (220, 220, 220),
        "text": (0, 0, 0), "corner": (150, 0, 0)
    }
    CELL_SIZE, MARGIN = 250, 2
    BOX_SIZE, GOAL_SIZE = 50, 80

    screen.fill(COLORS["background"])
    pygame.draw.rect(screen, (0, 0, 0), (MARGIN, MARGIN, 4 * CELL_SIZE, 2 * CELL_SIZE), width=5)
    robot_icon = pygame.image.load("robot_arm.png").convert_alpha()
    robot_icon = pygame.transform.scale(robot_icon, (50, 50))

    for row in range(2):
        for col in range(4):
            x = MARGIN + col * (CELL_SIZE + MARGIN)
            y = MARGIN + row * (CELL_SIZE + MARGIN)
            pygame.draw.rect(screen, COLORS["grid"], (x, y, CELL_SIZE, CELL_SIZE), width=2)
            screen.blit(robot_icon, (x + 10, y + 10))

    if isinstance(env, BoxNet2):
        for i in range(3):
            for j in range(5):
                cx = MARGIN + j * (CELL_SIZE + MARGIN) - MARGIN
                cy = MARGIN + i * (CELL_SIZE + MARGIN) - MARGIN
                pygame.draw.circle(screen, COLORS["corner"], (cx, cy), 6)

    for color, goal_list in env.goals.items():
        for (row, col) in goal_list:
            gx = MARGIN + col * (CELL_SIZE + MARGIN) + (CELL_SIZE - GOAL_SIZE) // 2
            gy = MARGIN + row * (CELL_SIZE + MARGIN) + (CELL_SIZE - GOAL_SIZE) // 2
            pygame.draw.rect(screen, (255, 255, 255), (gx, gy, GOAL_SIZE, GOAL_SIZE))
            pygame.draw.rect(screen, COLORS[color], (gx, gy, GOAL_SIZE, GOAL_SIZE), width=4)

    for box in env.boxes:
        positions = getattr(box, 'positions', [getattr(box, 'position', None)])
        for (row, col) in positions:
            if isinstance(env, BoxNet2):
                bx = MARGIN + col * (CELL_SIZE + MARGIN) - BOX_SIZE // 2
                by = MARGIN + row * (CELL_SIZE + MARGIN) - BOX_SIZE // 2
            else:
                cell_x = MARGIN + col * (CELL_SIZE + MARGIN)
                cell_y = MARGIN + row * (CELL_SIZE + MARGIN)
                bx = cell_x + (CELL_SIZE - BOX_SIZE) // 2
                by = cell_y + (CELL_SIZE - BOX_SIZE) // 2
            pygame.draw.rect(screen, COLORS[box.color], pygame.Rect(bx, by, BOX_SIZE, BOX_SIZE))

    pygame.display.flip()

# --- SIMULATION ---
def simulate_plan(env, actions, delay=1000):
    pygame.init()
    screen = pygame.display.set_mode((4 * 250 + 2 * 2, 2 * 250 + 2 * 2))
    pygame.display.set_caption("Simulation")

    print("Initial box states:")
    for b in env.boxes:
        print(f"- {b.color} at {getattr(b, 'positions', getattr(b, 'position', None))}")

    render_environment(screen, env)
    pygame.event.pump()
    pygame.time.wait(300)

    for step, (agent_id, color, from_pos, direction) in enumerate(actions):
        print(f"Step {step + 1}: Agent {agent_id} moves {color} box from {from_pos} {direction}")
        if color != "none":
            positions = lambda b: getattr(b, 'positions', [getattr(b, 'position', None)])
            box = next((b for b in env.boxes if b.color == color and from_pos in positions(b)), None)
            if box:
                moved = env.move_box(box, from_pos, direction)
                print(f"→ move_box() returned: {moved}")
            else:
                print(f"⚠️ Box not found: color={color}, from_pos={from_pos}")
        render_environment(screen, env)
        pygame.time.wait(delay)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    print("✅ Simulation complete.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

# --- MAIN TESTER ---
if __name__ == "__main__":
    env_classes = [BoxNet1, BoxNet2]
    for env_cls in env_classes:
        for planner_name, planner_fn in PLANNERS.items():
            print(f"\n==== {planner_name} on {env_cls.__name__} ====")
            plan_env = env_cls()
            if hasattr(plan_env, 'reset'):
                plan_env.reset()
            print("\n[PLANNING...]")
            plan_text, api_calls, _ = planner_fn(plan_env)
            if not plan_text:
                print("No valid plan returned.")
                continue
            print("\n[PLAN OUTPUT]\n")
            print(plan_text)
            actions = parse_llm_plan(plan_text)
            input("\nPress ENTER to start simulation...")
            sim_env = env_cls()
            simulate_plan(sim_env, actions, delay=300)
            input("\nPress ENTER to continue to next planner...")