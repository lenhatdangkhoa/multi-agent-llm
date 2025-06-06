import pygame
import time
import os
import re
from collections import defaultdict
import json
import argparse
import simulate_boxnet2
import pygame
import time

# Import environment models
from BoxNet1 import BoxNet1
from BoxNet2_test import BoxNet2

# Import planners
from CMAS import runCMAS
from HMAS1 import HMAS1
from HMAS2 import HMAS2
from DMAS import dmas_plan


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


def execute_plan_silently(env, actions):
    """Execute the plan without visual output to check validity."""
    for agent_id, color, from_pos, direction in actions:
        if color == "none":
            continue

        if direction == "goal":
            env.move_to_goal(color)
            continue

        # Find the box object
        box = next((b for b in env.boxes if b.color == color and from_pos in b.positions), None)
        if not box:
            return False

        success = env.move_box(box, from_pos, direction)
        if not success:
            return False

    return True


def run_planner(env, planner_name):
    """Run the specified planner on the environment."""
    if planner_name == "CMAS":
        from CMAS import format_prompt, call_llm
        prompt = format_prompt(env)
        response, tokens = call_llm(prompt)
        return response, 1, env
    elif planner_name == "DMAS":
        #from DMAS import run_dmas
        #plan_text, total_tokens = run_dmas(env)
        plan_text, api_calls = dmas_plan(env.boxes, env.goals)
        return plan_text, api_calls, env
    elif planner_name == "HMAS1":
        planner = HMAS1(environment_type="boxnet1" if isinstance(env, BoxNet1) else "boxnet2")
        planner.env = env
        plan, api_calls = planner.runHMAS1()
        return plan, api_calls, planner.env
    elif planner_name == "HMAS2":
        planner = HMAS2(environment_type="boxnet1" if isinstance(env, BoxNet1) else "boxnet2")
        planner.env = env
        plan, api_calls = planner.runHMAS2()
        return plan, api_calls, planner.env
    elif planner_name == "ETP":
        from ETP import intialPlan, call_llm, parse_llm_plan
        # Create a copy of the environment for planning
        if isinstance(env, BoxNet1):
            planning_env = BoxNet1()
        else:
            planning_env = BoxNet2()

        prompt = intialPlan(planning_env)
        response= call_llm(prompt)
        actions = parse_llm_plan(response)
        # Keep replanning until successful
        attempts = 0
        while not execute_plan_silently(planning_env, actions) and attempts < 3:
            attempts += 1
            prompt = intialPlan(planning_env)
            response += call_llm(prompt)
            actions += parse_llm_plan(response)
        return response, attempts + 1, env  # Return the final successful plan
    else:
        raise ValueError(f"Unknown planner: {planner_name}")



# Colors
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
        "red": (204, 0, 0), 
        "purple": (128, 0, 128),
        "green": (0, 204, 0)
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
    pygame.time.wait(3000)  # pause to show initial state

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


def main():
    parser = argparse.ArgumentParser(description="BoxNet Simulator")
    parser.add_argument("--env", choices=["boxnet1", "boxnet2"], default="boxnet2", help="Environment type")
    parser.add_argument("--planner", choices=["CMAS", "DMAS", "HMAS1", "HMAS2", "ETP"], default="HMAS2",
                        help="Planner type")
    parser.add_argument("--delay", type=int, default=500, help="Delay between steps (ms)")
    args = parser.parse_args()

    # Create environment
    if args.env == "boxnet1":
        env = BoxNet1()
    else:
        env = BoxNet2()
    
    print(f"Environment: {args.env}")
    # Run planner
    print(f"Running {args.planner} on {args.env}...")
    if args.planner == "DMAS":
        actions, api_calls, _ = dmas_plan(env, env.boxes, env.goals)
        if actions:
            if isinstance(env, BoxNet2):
                simulate_boxnet2.simulate_plan(env, actions, args.delay)
                exit()
            print(f"Simulating plan with {len(actions)} actions...")
            simulate_plan(env, actions, args.delay)
            return
    plan_text, api_calls, env = run_planner(env, args.planner)
    
    # Parse plan
    if plan_text is None:
        print("No valid plan was generated by the planner.")
        return

    actions = parse_llm_plan(plan_text)
    print(actions)
    # Simulate plan
    if isinstance(env, BoxNet2):
        simulate_boxnet2.simulate_plan(env, actions, args.delay)
        exit()
    if actions:
        print(f"Simulating plan with {len(actions)} actions...")
        simulate_plan(env, actions, args.delay)
    else:
        print("No valid plan to simulate.")


if __name__ == "__main__":
    main()
