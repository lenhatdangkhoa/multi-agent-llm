class Box:
    def __init__(self, color, positions):
        self.color = color
        self.positions = positions  

class Agent:
    def __init__(self, position):
        self.position = position


class BoxNet2:
    def __init__(self):
        self.GRID_WIDTH = 3
        self.GRID_HEIGHT = 5
        self.grid = [[0 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.boxes = [Box("blue", [(1,0)]), Box("yellow", [(1,3)]), Box("green", [(0,1)]), Box("purple", [(2,4)]), Box("red", [(1,2)])]
        self.goals = {
            "purple": [(0,0), (0,1), (1,0), (1,1)],
            "yellow": [(1,0), (2,0), (1,1), (2,1)],
            "blue": [(1,1), (1,2), (2,1), (2,2)],
            "green": [(1,3), (1,4), (2,3), (2,4)],
            "red": [(0,2), (0,3), (1,2), (1,3)]
        }        
        self.agents = [Agent([(0,0), (0,1), (1,0), (1,1)]), Agent([(0,1), (0,2), (1,1), (1,2)]), Agent([(0,2), (0,3), (1,2), (1,3)]), Agent([(0,3), (0,4), (1,3), (1,4)]),
                       Agent([(1,0), (1,1), (2,0), (2,1)]), Agent([(1,1), (1,2), (2,1), (2,2)]), Agent([(1,2), (1,3), (2,2), (2,3)]), Agent([(1,3), (1,4), (2,3), (2,4)])]

    def move_box(self, box, box_location, direction):
        change = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        new_x, new_y = box_location[0] + change[direction][0], box_location[1] + change[direction][1]
    
        if new_x < 0 or new_x >= self.GRID_WIDTH or new_y < 0 or new_y >= self.GRID_HEIGHT:
            print("Invalid move")
            return False
        
        if box_location in box.positions:
            box.positions.remove(box_location)
            box.positions.append((new_x, new_y))
            print(f"{box.color} box moved to {(new_x, new_y)}")
            return True
        else:
            print("Box not in position")
            return False
        
    def move_to_goal(self, color):
        self.goals[color] = []
        for box in self.boxes:
            if box.color == color:
                self.boxes.remove(box)


        


        
    