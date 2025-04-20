import re
import BoxNet1
from collections import defaultdict

def parse_llm_plan(text):
    actions = []

    pattern_move = r"- Agent (\d+): move (\w+) box from \((\d+), (\d+)\) to \((\d+), (\d+)\) \[(\w+)\]"
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


actions = parse_llm_plan("""- Agent 0: move blue box from (0, 0) to (1, 0) [down]
- Agent 4: move yellow box from (0, 1) to (1, 1) [down]
- Agent 5: move yellow box from (0, 3) to (1, 3) [down]
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

import pygame
import time

# Colors
COLORS = {
    "blue": (0, 102, 204),
    "yellow": (255, 204, 0),
    "red": (204, 0, 0),
    "goal": (200, 200, 200),
    "agent": (50, 205, 50),
    "background": (255, 255, 255),
    "grid": (180, 180, 180),
    "text": (0, 0, 0)
}

CELL_SIZE = 250
MARGIN = 2
GRID_WIDTH = 4 * CELL_SIZE
GRID_HEIGHT = 2 * CELL_SIZE

def render_environment(screen, env):
    
    # Background
    screen.fill((255, 255, 255))
    pygame.draw.rect(screen, (0,0,0), (MARGIN, MARGIN, GRID_WIDTH, GRID_HEIGHT), width=5)
    # Draw grid
    robot_icon = pygame.image.load("robot_arm.png").convert_alpha()  # preserves transparency
    robot_icon = pygame.transform.scale(robot_icon, (50, 50))    # resize to fit corner
    for row in range(2):
        for col in range(4):
            x = MARGIN + col * (CELL_SIZE + MARGIN)
            y = MARGIN + row * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (0, 0, 0), rect, width=2)
            screen.blit(robot_icon, (x + 10, y + 10)) 
    
    # Draw goals
    GOAL_SIZE = 80  # size of goal square
    GOAL_COLORS = {
        "blue": (0, 102, 204),
        "yellow": (255, 204, 0),
        "red": (204, 0, 0)
    }
    for color, goal_list in env.goals.items():
        for (row, col) in goal_list:
            gx = MARGIN + col * (CELL_SIZE + MARGIN) + (CELL_SIZE - GOAL_SIZE) // 2
            gy = MARGIN + row * (CELL_SIZE + MARGIN) + (CELL_SIZE - GOAL_SIZE) // 2
            goal_rect = pygame.Rect(gx, gy, GOAL_SIZE, GOAL_SIZE)

            pygame.draw.rect(screen, (255, 255, 255), goal_rect)  # white fill
            pygame.draw.rect(screen, GOAL_COLORS[color], goal_rect, width=4)  # colored border
    
    BOX_SIZE = 50
    # Draw boxes
    for box in env.boxes:
        for (row, col) in box.positions:
            cell_x = MARGIN + col * (CELL_SIZE + MARGIN)
            cell_y = MARGIN + row * (CELL_SIZE + MARGIN)
            bx = cell_x + (CELL_SIZE - BOX_SIZE) // 2
            by = cell_y + (CELL_SIZE - BOX_SIZE) // 2
            box_rect = pygame.Rect(bx, by, BOX_SIZE, BOX_SIZE)
            pygame.draw.rect(screen, COLORS[box.color], box_rect)

    font = pygame.font.SysFont("Arial", 20)
    box_counts = defaultdict(int)

    # Count boxes at each cell (row, col)
    for box in env.boxes:
        for (row, col) in box.positions:
            box_counts[(row, col)] += 1
    for (row, col), count in box_counts.items():
        x = MARGIN + col * (CELL_SIZE + MARGIN)
        y = MARGIN + row * (CELL_SIZE + MARGIN)
        text = font.render((f"Box Count: {count}"), True, (0, 0, 0))  # black text
        screen.blit(text, (x + CELL_SIZE - 150, y + 5))  # top-right corner of cell

    pygame.display.flip()

def simulate_plan(env, actions, delay=1000):
    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH + 2 * MARGIN, GRID_HEIGHT + 2 * MARGIN))
    pygame.display.set_caption("BoxNet1 Simulation")

    render_environment(screen, env)
    pygame.event.pump()
    pygame.time.wait(1000)  # pause to show initial state

    for step, (agent_id, color, from_pos, direction) in enumerate(actions):
        print(f"Step {step + 1}: Agent {agent_id} moves {color} box from {from_pos} {direction}")

        # Find the box object
        if color != "none":
            box = next((b for b in env.boxes if b.color == color and from_pos in b.positions), None)
            if box:
                env.move_box(box, from_pos, direction)

        render_environment(screen, env)
        pygame.time.wait(delay)  # wait between steps

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    print("✅ Simulation complete.")
    # Keep final screen open until closed
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()


env = BoxNet1.BoxNet1()
simulate_plan(env, actions, delay=1000)