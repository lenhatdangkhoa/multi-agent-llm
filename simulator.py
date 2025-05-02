import pygame
import time
import os
import re
from collections import defaultdict
import json
import argparse

# Import environment models
from BoxNet1 import BoxNet1
from BoxNet2_test import BoxNet2

# Import planners
from CMAS import runCMAS
from DMAS import run_dmas
from HMAS1 import HMAS1
from HMAS2 import HMAS2
from ETP import runETP

# Define colors
COLORS = {
    "blue": (0, 102, 204),
    "yellow": (255, 204, 0),
    "red": (204, 0, 0),
    "green": (0, 153, 0),
    "purple": (153, 0, 153),
    "goal": (200, 200, 200),
    "agent": (50, 205, 50),
    "background": (255, 255, 255),
    "grid": (220, 220, 220),
    "text": (0, 0, 0),
    "corner": (150, 0, 0)
}


def parse_llm_plan(text):
    """Parse LLM output into actionable steps with improved multi-step direction handling."""
    actions = []

    if isinstance(text, dict):
        lines = [f"Agent {k.replace('Agent', '')}: {v}" for k, v in text.items()]
    else:
        lines = text.strip().split('\n')
        lines = [line for line in lines if line.strip() and line.strip().lower() != "plan:"]

    # Direction mapping
    dir_map = {"north": "up", "south": "down", "east": "right", "west": "left"}

    # Regex patterns for different action formats
    pattern_move = r"-? ?Agent (\d+): move (\w+) box from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)(?:\s*\[?(\w+(?:,\s*\w+)*)\]?)?"
    pattern_nothing = r"-? ?Agent (\d+): do nothing"
    pattern_move_to_goal = r"-? ?Agent (\d+): move (\w+) box to goal"
    pattern_standby = r"-? ?Agent (\d+): [Ss]tandby"

    for line in lines:
        line = line.strip()
        move_match = re.match(pattern_move, line)
        nothing_match = re.match(pattern_nothing, line)
        move_to_goal_match = re.match(pattern_move_to_goal, line)
        standby_match = re.match(pattern_standby, line)

        if move_match:
            agent_id = int(move_match.group(1))
            color = move_match.group(2)
            from_pos = (int(move_match.group(3)), int(move_match.group(4)))

            # Handle multi-step directions
            if move_match.group(7):
                directions = [d.strip() for d in move_match.group(7).split(',')]
                # Create separate actions for each direction
                current_pos = from_pos
                for direction in directions:
                    # Map direction name if needed
                    direction = dir_map.get(direction.lower(), direction.lower())

                    # Calculate next position based on direction
                    if direction == "up":
                        next_pos = (current_pos[0] - 1, current_pos[1])
                    elif direction == "down":
                        next_pos = (current_pos[0] + 1, current_pos[1])
                    elif direction == "left":
                        next_pos = (current_pos[0], current_pos[1] - 1)
                    elif direction == "right":
                        next_pos = (current_pos[0], current_pos[1] + 1)
                    else:
                        print(f"⚠️ Unknown direction: {direction}")
                        continue

                    actions.append((agent_id, color, current_pos, direction))
                    current_pos = next_pos
            else:
                # Single direction
                raw_dir = move_match.group(7).lower() if move_match.group(7) else ""
                direction = dir_map.get(raw_dir, raw_dir)
                actions.append((agent_id, color, from_pos, direction))

        elif nothing_match or standby_match:
            agent_id = int((nothing_match or standby_match).group(1))
            actions.append((agent_id, "none", None, "stay"))

        elif move_to_goal_match:
            agent_id = int(move_to_goal_match.group(1))
            color = move_to_goal_match.group(2)
            actions.append((agent_id, color, None, "goal"))

        else:
            print(f"⚠️ Unrecognized plan line skipped: {line}")

    return actions


def render_environment(screen, env, step=0, total_steps=0):
    """Render the current state of the environment."""
    CELL_SIZE, MARGIN = 150, 5
    BOX_SIZE, GOAL_SIZE = 40, 60
    FONT_SIZE = 24

    # Clear screen
    screen.fill(COLORS["background"])

    # Get grid dimensions
    grid_width = env.GRID_WIDTH
    grid_height = env.GRID_HEIGHT

    # Draw grid
    for row in range(grid_height):
        for col in range(grid_width):
            x = MARGIN + col * (CELL_SIZE + MARGIN)
            y = MARGIN + row * (CELL_SIZE + MARGIN)
            pygame.draw.rect(screen, COLORS["grid"], (x, y, CELL_SIZE, CELL_SIZE), width=2)

    # Draw goals
    for color, goal_list in env.goals.items():
        for goal_pos in goal_list:
            gx = MARGIN + goal_pos[1] * (CELL_SIZE + MARGIN) + (CELL_SIZE - GOAL_SIZE) // 2
            gy = MARGIN + goal_pos[0] * (CELL_SIZE + MARGIN) + (CELL_SIZE - GOAL_SIZE) // 2
            pygame.draw.rect(screen, (255, 255, 255), (gx, gy, GOAL_SIZE, GOAL_SIZE))
            pygame.draw.rect(screen, COLORS[color], (gx, gy, GOAL_SIZE, GOAL_SIZE), width=4)

    # Draw agents
    for i, agent in enumerate(env.agents):
        pos = getattr(agent, 'position', None)
        if pos is None and hasattr(agent, 'cell'):
            # For BoxNet2, draw agent at center of its cell
            cell_x = sum(c[1] for c in agent.cell) / len(agent.cell)
            cell_y = sum(c[0] for c in agent.cell) / len(agent.cell)
            ax = MARGIN + cell_x * (CELL_SIZE + MARGIN) + 10
            ay = MARGIN + cell_y * (CELL_SIZE + MARGIN) + 10
        elif pos:
            ax = MARGIN + pos[1] * (CELL_SIZE + MARGIN) + 10
            ay = MARGIN + pos[0] * (CELL_SIZE + MARGIN) + 10
        else:
            continue

        pygame.draw.circle(screen, COLORS["agent"], (ax, ay), 15)

        # Draw agent ID
        font = pygame.font.SysFont('Arial', FONT_SIZE)
        text = font.render(str(i), True, COLORS["text"])
        screen.blit(text, (ax - 5, ay - 10))

    # Draw boxes
    for box in env.boxes:
        positions = getattr(box, 'positions', [])
        for pos in positions:
            bx = MARGIN + pos[1] * (CELL_SIZE + MARGIN) + (CELL_SIZE - BOX_SIZE) // 2
            by = MARGIN + pos[0] * (CELL_SIZE + MARGIN) + (CELL_SIZE - BOX_SIZE) // 2
            pygame.draw.rect(screen, COLORS[box.color], (bx, by, BOX_SIZE, BOX_SIZE))

    # Draw corners for BoxNet2
    if isinstance(env, BoxNet2):
        for agent in env.agents:
            for corner in agent.cell:
                cx = MARGIN + corner[1] * (CELL_SIZE + MARGIN)
                cy = MARGIN + corner[0] * (CELL_SIZE + MARGIN)
                pygame.draw.circle(screen, COLORS["corner"], (cx, cy), 6)

    # Draw step counter
    font = pygame.font.SysFont('Arial', FONT_SIZE)
    step_text = font.render(f"Step: {step}/{total_steps}", True, COLORS["text"])
    screen.blit(step_text, (10, grid_height * (CELL_SIZE + MARGIN) + 10))

    pygame.display.flip()


def simulate_plan(env, actions, delay=500):
    """Simulate the execution of a plan with state tracking."""
    pygame.init()

    # Calculate window size based on grid dimensions
    window_width = (env.GRID_WIDTH * 150) + ((env.GRID_WIDTH + 1) * 5)
    window_height = (env.GRID_HEIGHT * 150) + ((env.GRID_HEIGHT + 1) * 5) + 50  # Extra space for text

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("BoxNet Simulation")

    # Dictionary to track current positions of boxes
    box_positions = {}
    for box in env.boxes:
        if hasattr(box, 'positions') and box.positions:
            box_positions[box.color] = box.positions[0]
        elif hasattr(box, 'position'):
            box_positions[box.color] = box.position

    # Print initial state
    print("Initial box states:")
    for b in env.boxes:
        print(f"- {b.color} at {getattr(b, 'positions', getattr(b, 'position', None))}")

    # Render initial state
    render_environment(screen, env, 0, len(actions))
    pygame.time.wait(delay)

    # Execute each action
    for step, (agent_id, color, from_pos, direction) in enumerate(actions):
        print(f"Step {step + 1}: Agent {agent_id} moves {color} box from {from_pos} {direction}")

        if color != "none":
            # Use tracked position instead of plan's position if there's a mismatch
            if from_pos is not None and color in box_positions:
                actual_pos = box_positions[color]
                if actual_pos != from_pos:
                    print(f"⚠️ Position mismatch: Plan says {from_pos}, but box is at {actual_pos}")
                    from_pos = actual_pos

            if direction == "goal":
                env.move_to_goal(color)
                print(f"→ Box moved to goal")
                # Remove from tracking
                if color in box_positions:
                    del box_positions[color]
            else:
                # Find the box object
                box = None
                for b in env.boxes:
                    if b.color == color:
                        positions = getattr(b, 'positions', [getattr(b, 'position', None)])
                        if from_pos in positions:
                            box = b
                            break

                if box:
                    moved = env.move_box(box, from_pos, direction)
                    print(f"→ move_box() returned: {moved}")

                    if moved:
                        # Update tracked position
                        if hasattr(box, 'positions'):
                            for pos in box.positions:
                                if pos != from_pos:
                                    box_positions[color] = pos
                                    break
                        elif hasattr(box, 'position'):
                            box_positions[color] = box.position
                else:
                    print(f"⚠️ Box not found: color={color}, from_pos={from_pos}")

        # Render updated state
        render_environment(screen, env, step + 1, len(actions))
        pygame.time.wait(delay)

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    print("✅ Simulation complete.")

    # Wait for user to close window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


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
        response = call_llm(prompt)
        return response.choices[0].message.content, 1, env
    elif planner_name == "DMAS":
        from DMAS import run_dmas
        plan_text, total_tokens = run_dmas(env)
        return plan_text, total_tokens, env
    elif planner_name == "HMAS1":
        planner = HMAS1(environment_type="boxnet1" if isinstance(env, BoxNet1) else "boxnet2")
        planner.env = env
        plan = planner.run_planning()
        return plan, planner.token_count, planner.env
    elif planner_name == "HMAS2":
        planner = HMAS2(environment_type="boxnet1" if isinstance(env, BoxNet1) else "boxnet2")
        planner.env = env
        plan = planner.run_planning()
        return plan, planner.token_count, planner.env
    elif planner_name == "ETP":
        from ETP import intialPlan, call_llm, parse_llm_plan

        # Create a copy of the environment for planning
        if isinstance(env, BoxNet1):
            planning_env = BoxNet1()
        else:
            planning_env = BoxNet2()

        prompt = intialPlan(planning_env)
        response = call_llm(prompt)
        actions = parse_llm_plan(response)

        # Keep replanning until successful
        attempts = 0
        while not execute_plan_silently(planning_env, actions) and attempts < 3:
            attempts += 1
            prompt = intialPlan(planning_env)
            response = call_llm(prompt)
            actions = parse_llm_plan(response)

        return response, attempts + 1, env  # Return the final successful plan
    else:
        raise ValueError(f"Unknown planner: {planner_name}")


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

    # Run planner
    print(f"Running {args.planner} on {args.env}...")
    plan_text, api_calls, env = run_planner(env, args.planner)

    # Parse plan
    actions = parse_llm_plan(plan_text)

    # Simulate plan
    if actions:
        print(f"Simulating plan with {len(actions)} actions...")
        simulate_plan(env, actions, args.delay)
    else:
        print("No valid plan to simulate.")


if __name__ == "__main__":
    main()
