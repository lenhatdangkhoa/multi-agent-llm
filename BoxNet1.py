class Box:
    def __init__(self, color, positions):
        self.color = color
        self.positions = positions  


class Agent:
    def __init__(self, position):
        self.position = position


class BoxNet1:
    def __init__(self):
        self.GRID_WIDTH = 2
        self.GRID_HEIGHT = 4
        self.grid = [[0 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.boxes = [Box("blue", [(0,0)]), Box("yellow", [(0,1), (0,3)]), Box("red", [(1,2), (1,2)])]
        self.goals = {
            "blue": [(1,1)],
            "yellow": [(1,0), (1,3)],
            "red": [(0,0), (0,2)]
        }        
        self.agents = [Agent((0,0)), Agent((0,1)), Agent((0,2)), Agent((0,3)), Agent((1,0)), Agent((1,1)), Agent((1,2)), Agent((1,3))]

    def move_box(self, box, box_location, direction):
        change = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        if direction not in change.keys():
            return False
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
        


        
    