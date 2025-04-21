import re
temp = "ACTION: move blue box from (0, 0) to cell (1, 0)"
parts = temp.split()
color = parts[2].strip("[]")
print(f"Color: {color}")
match = re.search(r"\((\d+),\s*(\d+)\)\sto cell\s*\((\d+),\s*(\d+)\)", temp)
if match:
    box_row = int(match.group(1))
    box_col = int(match.group(2))
    print(f"Box Row: {box_row}, Box Col: {box_col}")
    row, col = int(match.group(3)), int(match.group(4))
    print(f"Row: {row}, Col: {col}")