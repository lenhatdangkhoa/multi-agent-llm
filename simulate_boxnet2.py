import pygame
from collections import defaultdict
import BoxNet2_test
# === Constants ===
COLORS = {
    "blue": (0, 102, 204),
    "yellow": (255, 204, 0),
    "red": (204, 0, 0),
    "purple": (128, 0, 128),
    "green": (0, 204, 0),
    "goal": (200, 200, 200),
    "agent": (50, 205, 50),
    "background": (255, 255, 255),
    "grid": (180, 180, 180),
    "text": (0, 0, 0)
}
CELL_SIZE = 180
MARGIN = 2
GRID_COLS = 5
GRID_ROWS = 3
GRID_WIDTH = GRID_COLS * CELL_SIZE
GRID_HEIGHT = GRID_ROWS * CELL_SIZE


def render_environment(screen, env):
    screen.fill(COLORS["background"])
    pygame.draw.rect(screen, COLORS["grid"], (MARGIN, MARGIN, GRID_WIDTH, GRID_HEIGHT), width=4)

    robot_icon = pygame.image.load("robot_arm.png").convert_alpha()
    robot_icon = pygame.transform.scale(robot_icon, (40, 40))

    # Draw cells and agent icons
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = MARGIN + col * (CELL_SIZE + MARGIN)
            y = MARGIN + row * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLORS["grid"], rect, width=2)

    # Draw robot icons in agent corner cells
    # Draw robot icon at the center of the agent's 2x2 region
    for i, agent in enumerate(env.agents):
        corners = agent.position
        avg_row = sum(pos[0] for pos in corners) / 4
        avg_col = sum(pos[1] for pos in corners) / 4

        center_x = MARGIN + avg_col * (CELL_SIZE + MARGIN) + (CELL_SIZE - robot_icon.get_width()) // 2
        center_y = MARGIN + avg_row * (CELL_SIZE + MARGIN) + (CELL_SIZE - robot_icon.get_height()) // 2
        screen.blit(robot_icon, (center_x, center_y))

    # Group goals by (row, col)
    goal_cells = defaultdict(list)
    for color, positions in env.goals.items():
        for (row, col) in positions:
            goal_cells[(row, col)].append(color)

    # Draw multiple goal overlays per cell
    GOAL_SIZE = 50
    for (row, col), color_list in goal_cells.items():
        num_goals = len(color_list)
        cell_x = MARGIN + col * (CELL_SIZE + MARGIN)
        cell_y = MARGIN + row * (CELL_SIZE + MARGIN)

        padding = 5
        spacing = GOAL_SIZE + padding
        total_width = spacing * num_goals - padding
        start_x = cell_x + (CELL_SIZE - total_width) // 2
        center_y = cell_y + (CELL_SIZE - GOAL_SIZE) // 2

        for i, color in enumerate(color_list):
            gx = start_x + i * spacing
            gy = center_y
            goal_rect = pygame.Rect(gx, gy, GOAL_SIZE, GOAL_SIZE)
            pygame.draw.rect(screen, COLORS["background"], goal_rect)
            pygame.draw.rect(screen, COLORS[color], goal_rect, width=4)


    # Draw boxes
    BOX_SIZE = 40
    for box in env.boxes:
        for (row, col) in box.positions:
            x = MARGIN + col * (CELL_SIZE + MARGIN) + (CELL_SIZE - BOX_SIZE) // 2
            y = MARGIN + row * (CELL_SIZE + MARGIN) + (CELL_SIZE - BOX_SIZE) // 2
            box_rect = pygame.Rect(x, y, BOX_SIZE, BOX_SIZE)
            pygame.draw.rect(screen, COLORS[box.color], box_rect)

    # Draw box counts
    font = pygame.font.SysFont("Arial", 16)
    box_counts = defaultdict(int)
    for box in env.boxes:
        for pos in box.positions:
            box_counts[pos] += 1

    for (row, col), count in box_counts.items():
        x = MARGIN + col * (CELL_SIZE + MARGIN)
        y = MARGIN + row * (CELL_SIZE + MARGIN)
        label = font.render(f"{count}", True, COLORS["text"])
        screen.blit(label, (x + CELL_SIZE - 20, y + 5))

    pygame.display.flip()


def simulate_plan(env, actions, delay=500):
    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH + 2 * MARGIN, GRID_HEIGHT + 2 * MARGIN))
    pygame.display.set_caption("BoxNet2 Simulation")

    render_environment(screen, env)
    pygame.time.wait(2000)

    for step, (agent_id, color, from_pos, direction) in enumerate(actions):
        print(f"Step {step + 1}: Agent {agent_id} moves {color} box from {from_pos} {direction}")
        if color != "none":
            box = next((b for b in env.boxes if b.color == color and from_pos in b.positions), None)
            if box:
                env.move_box(box, from_pos, direction)
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
def main():
    env = BoxNet2_test.BoxNet2()
    actions = [(4, 'blue', (1, 0), 'right'), (5, 'blue', (1, 1), 'right'), (6, 'blue', (1, 2), 'down'), (1, 'red', (1, 2), 'left'), (0, 'green', (0, 1), 'down'), (4, 'yellow', (1, 3), 'left'), (3, 'green', (1, 0), 'right'), (0, 'purple', (2, 4), 'up'), (1, 'purple', (1, 1), 'up'), (0, 'purple', (0, 1), 'left'), (0, 'purple', None, 'goal'), (1, 'red', None, 'goal'), (2, 'none', None, 'stay'), (3, 'green', None, 'goal'), (4, 'none', None, 'stay'), (5, 'blue', None, 'goal'), (6, 'none', None, 'stay'), (7, 'none', None, 'stay'), (4, 'blue', (1, 0), 'right'), (5, 'blue', (1, 1), 'right'), (6, 'blue', (1, 2), 'down'), (1, 'red', (1, 2), 'left'), (0, 'green', (0, 1), 'down'), (4, 'yellow', (1, 3), 'left'), (3, 'green', (1, 0), 'right'), (0, 'purple', (2, 4), 'up'), (1, 'purple', (1, 1), 'up'), (0, 'purple', (0, 1), 'left'), (0, 'purple', None, 'goal'), (1, 'red', None, 'goal'), (2, 'none', None, 'stay'), (3, 'green', None, 'goal'), (4, 'none', None, 'stay'), (5, 'blue', None, 'goal'), (6, 'none', None, 'stay'), (1, 'green', (0, 1), 'down'), (3, 'green', None, 'goal'), (6, 'yellow', (1, 3), 'left'), (4, 'yellow', None, 'goal'), (2, 'red', (1, 2), 'up'), (1, 'red', None, 'goal'), (7, 'purple', (2, 4), 'up'), (0, 'purple', None, 'goal'), (4, 'blue', (1, 0), 'right'), (5, 'blue', (1, 1), 'right'), (6, 'blue', (1, 2), 'down'), (1, 'red', (1, 2), 'left'), (0, 'green', (0, 1), 'down'), (4, 'yellow', (1, 3), 'left'), (3, 'green', (1, 0), 'right'), (0, 'purple', (2, 4), 'up'), (1, 'purple', (1, 1), 'up'), (0, 'purple', (0, 1), 'left'), (0, 'purple', None, 'goal'), (1, 'red', None, 'goal'), (2, 'none', None, 'stay'), (3, 'green', None, 'goal'), (4, 'none', None, 'stay'), (5, 'blue', None, 'goal'), (6, 'none', None, 'stay'), (1, 'green', (0, 1), 'down'), (3, 'green', None, 'goal'), (6, 'yellow', (1, 3), 'left'), (4, 'yellow', None, 'goal'), (2, 'red', (1, 2), 'up'), (1, 'red', None, 'goal'), (7, 'purple', (2, 4), 'up'), (1, 'green', (0, 1), 'down'), (3, 'green', None, 'goal'), (6, 'yellow', (1, 3), 'left'), (4, 'yellow', None, 'goal'), (2, 'red', (1, 2), 'up'), (1, 'red', None, 'goal'), (7, 'purple', (2, 4), 'up'), (0, 'purple', None, 'goal'), (4, 'blue', (1, 0), 'right'), (5, 'blue', (1, 1), 'right'), (6, 'blue', (1, 2), 'down'), (1, 'red', (1, 2), 'left'), (0, 'green', (0, 1), 'down'), (4, 'yellow', (1, 3), 'left'), (3, 'green', (1, 0), 'right'), (0, 'purple', (2, 4), 'up'), (1, 'purple', (1, 1), 'up'), (0, 'purple', (0, 1), 'left'), (0, 'purple', None, 'goal'), (1, 'red', None, 'goal'), (2, 'none', None, 'stay'), (3, 'green', None, 'goal'), (4, 'none', None, 'stay'), (5, 'blue', None, 'goal'), (6, 'none', None, 'stay'), (1, 'green', (0, 1), 'down'), (3, 'green', None, 'goal'), (6, 'yellow', (1, 3), 'left'), (4, 'yellow', None, 'goal'), (2, 'red', (1, 2), 'up'), (1, 'red', None, 'goal'), (7, 'purple', (2, 4), 'up'), (1, 'green', (0, 1), 'down'), (3, 'green', None, 'goal'), (6, 'yellow', (1, 3), 'left'), (4, 'yellow', None, 'goal'), (2, 'red', (1, 2), 'up'), (1, 'red', None, 'goal'), (7, 'purple', (2, 4), 'up'), (1, 'green', (0, 1), 'down'), (3, 'green', None, 'goal'), (6, 'yellow', (1, 3), 'left'), (4, 'yellow', None, 'goal'), (2, 'red', (1, 2), 'up'), (1, 'red', None, 'goal'), (7, 'purple', (2, 4), 'up'), (0, 'purple', None, 'goal'), (0, 'none', None, 'stay'), (5, 'none', None, 'stay')]
    simulate_plan(env, actions)

if __name__ == "__main__":
    main()