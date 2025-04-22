class Box:
    def __init__(self, color, position=None):
        self.color = color
        self.position = position  # Can be a corner position or a cell position
        self.at_goal = False


class Corner:
    def __init__(self, position, connected_cells):
        self.position = position  # (x, y, corner_id) where corner_id is NE, NW, SE, SW
        self.connected_cells = connected_cells  # List of cell coordinates this corner connects
        self.occupied_by = None  # Box occupying this corner, None if empty


class Agent:
    def __init__(self, cell_position):
        self.cell_position = cell_position  # (x, y) of the cell the agent is confined to

    def get_available_actions(self, environment):
        """Returns list of available actions for this agent based on current environment state"""
        actions = []
        cell_x, cell_y = self.cell_position

        # Get corners in this cell
        cell_corners = []
        for corner in environment.corners:
            if (cell_x, cell_y) in corner.connected_cells:
                cell_corners.append(corner)

        # Get boxes at corners in this cell
        for corner in cell_corners:
            if corner.occupied_by:
                # Action 1: Move box from this corner to another corner in the cell
                for target_corner in cell_corners:
                    if target_corner != corner and not target_corner.occupied_by:
                        actions.append(
                            f"move_box_corner_to_corner({corner.occupied_by.color}, {target_corner.position})")

                # Action 2: Move box from corner to goal if color matches
                box_color = corner.occupied_by.color
                if box_color in environment.goals:
                    for goal_pos in environment.goals[box_color]:
                        if goal_pos[0] == cell_x and goal_pos[1] == cell_y:
                            actions.append(f"move_box_corner_to_goal({box_color}, {goal_pos})")

        # Action 3: Do nothing
        actions.append("do_nothing()")

        return actions


class BoxNet2:
    def __init__(self, grid_width=2, grid_height=2):
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Initialize corners
        self.corners = []
        for x in range(grid_width + 1):
            for y in range(grid_height + 1):
                # Each corner connects to up to 4 cells
                connected_cells = []
                for dx, dy, corner_id in [(0, 0, "SE"), (-1, 0, "SW"), (0, -1, "NE"), (-1, -1, "NW")]:
                    cell_x, cell_y = x + dx, y + dy
                    if 0 <= cell_x < grid_width and 0 <= cell_y < grid_height:
                        connected_cells.append((cell_x, cell_y))

                if connected_cells:  # Only create corner if it connects to at least one cell
                    for dx, dy, corner_id in [(0, 0, "SE"), (-1, 0, "SW"), (0, -1, "NE"), (-1, -1, "NW")]:
                        cell_x, cell_y = x + dx, y + dy
                        if 0 <= cell_x < grid_width and 0 <= cell_y < grid_height:
                            self.corners.append(Corner((x, y, corner_id), connected_cells))

        # Initialize boxes, goals, and agents (to be set by the scenario)
        self.boxes = []
        self.goals = {}
        self.agents = []

    def place_box_at_corner(self, box, corner_position):
        """Place a box at a specified corner"""
        # Find the corner
        target_corner = None
        for corner in self.corners:
            if corner.position == corner_position:
                target_corner = corner
                break

        if not target_corner:
            print(f"Corner {corner_position} not found")
            return False

        if target_corner.occupied_by:
            print(f"Corner {corner_position} already occupied by {target_corner.occupied_by.color} box")
            return False

        # Place the box
        target_corner.occupied_by = box
        box.position = corner_position
        print(f"{box.color} box placed at corner {corner_position}")
        return True

    def move_box_corner_to_corner(self, agent, box_color, target_corner_position):
        """Move a box from one corner to another within the same cell"""
        agent_x, agent_y = agent.cell_position

        # Find the source corner with the box
        source_corner = None
        for corner in self.corners:
            if (agent_x,
                agent_y) in corner.connected_cells and corner.occupied_by and corner.occupied_by.color == box_color:
                source_corner = corner
                break

        if not source_corner:
            print(f"No {box_color} box found at any corner in cell ({agent_x}, {agent_y})")
            return False

        # Find the target corner
        target_corner = None
        for corner in self.corners:
            if corner.position == target_corner_position and (agent_x, agent_y) in corner.connected_cells:
                target_corner = corner
                break

        if not target_corner:
            print(
                f"Target corner {target_corner_position} not found or not accessible from cell ({agent_x}, {agent_y})")
            return False

        if target_corner.occupied_by:
            print(f"Target corner {target_corner_position} already occupied by {target_corner.occupied_by.color} box")
            return False

        # Move the box
        box = source_corner.occupied_by
        source_corner.occupied_by = None
        target_corner.occupied_by = box
        box.position = target_corner_position
        print(f"{box_color} box moved from {source_corner.position} to {target_corner_position}")
        return True

    def move_box_corner_to_goal(self, agent, box_color, goal_position):
        """Move a box from a corner to a goal location within the same cell"""
        agent_x, agent_y = agent.cell_position

        # Check if goal position is in the agent's cell
        if goal_position[0] != agent_x or goal_position[1] != agent_y:
            print(f"Goal position {goal_position} not in agent's cell ({agent_x}, {agent_y})")
            return False

        # Check if goal is for the right color
        if box_color not in self.goals or goal_position not in self.goals[box_color]:
            print(f"No {box_color} goal at position {goal_position}")
            return False

        # Find the source corner with the box
        source_corner = None
        for corner in self.corners:
            if (agent_x,
                agent_y) in corner.connected_cells and corner.occupied_by and corner.occupied_by.color == box_color:
                source_corner = corner
                break

        if not source_corner:
            print(f"No {box_color} box found at any corner in cell ({agent_x}, {agent_y})")
            return False

        # Move the box to the goal
        box = source_corner.occupied_by
        source_corner.occupied_by = None
        box.position = goal_position
        box.at_goal = True
        print(f"{box_color} box moved from {source_corner.position} to goal at {goal_position}")
        return True

    def do_nothing(self, agent):
        """Agent does nothing this turn"""
        print(f"Agent at {agent.cell_position} does nothing")
        return True

    def check_task_completion(self):
        """Check if all boxes are at their goals"""
        for box in self.boxes:
            if not box.at_goal:
                return False
        return True

    def get_corner_occupancy(self):
        """Returns a dictionary mapping corner positions to the boxes occupying them"""
        occupancy = {}
        for corner in self.corners:
            occupancy[corner.position] = corner.occupied_by.color if corner.occupied_by else "EMPTY"
        return occupancy

    def get_goal_status(self):
        """Returns a dictionary showing which goals are met and which are unmet"""
        status = {}
        for color, goal_positions in self.goals.items():
            for goal_pos in goal_positions:
                # Check if any box of this color is at this goal
                goal_met = False
                for box in self.boxes:
                    if box.color == color and box.position == goal_pos and box.at_goal:
                        goal_met = True
                        status[f"{color}_{goal_pos}"] = f"met at goal{goal_pos}"
                        break
                if not goal_met:
                    status[f"{color}_{goal_pos}"] = f"unmet at goal{goal_pos}"
        return status

    def setup_scenario(self, boxes_data, goals_data, agents_data):
        """Set up a specific scenario with boxes, goals, and agents"""
        # Create boxes
        self.boxes = []
        for color, corner_pos in boxes_data:
            box = Box(color)
            self.boxes.append(box)
            if corner_pos:  # If initial position is provided
                self.place_box_at_corner(box, corner_pos)

        # Set goals
        self.goals = goals_data

        # Create agents
        self.agents = []
        for cell_pos in agents_data:
            self.agents.append(Agent(cell_pos))

    def get_environment_state(self):
        """Returns a dictionary representation of the current environment state"""
        state = {}

        # Add cell contents (for visualization)
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                cell_key = f"{x}_{y}"
                state[cell_key] = []

                # Add goals in this cell
                for color, positions in self.goals.items():
                    for pos in positions:
                        if pos[0] == x and pos[1] == y:
                            state[cell_key].append(f"target_{color}")

        # Add corner occupancy
        corner_occupancy = self.get_corner_occupancy()
        state["corners"] = corner_occupancy

        # Add box positions
        box_locations = {}
        for box in self.boxes:
            if box.position:
                if isinstance(box.position, tuple) and len(box.position) == 3:  # Corner position
                    box_locations[f"box_{box.color}"] = f"corner{box.position}"
                else:  # Goal position
                    box_locations[f"box_{box.color}"] = f"cell{box.position}"
        state["boxes"] = box_locations

        return state
