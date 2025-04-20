import BoxNet1
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def format_prompt(env: BoxNet1):
    lines = [
        "You are a centralized task planner for a grid-based environment.",
        "The grid is represented as a 2D array, where each cell can contain a box or an agent.",
        "Your job is to assign each agent to move a colored box to its goal.",
        "If the colored box has multiple goals, the agent should move it to any goal. Make sure all the colored boxes are moved to their goals.",
        "Each agent can only move boxes that are within its own grid cell.",
        "Each agent is stuck in its own cell and can only move boxes to adjacent cells.",
        "Multiple boxes can occupy the same cell.",
        "There are three possible actions for each agent: move box to an ajacent cell (up, down, left, right),"
        "move box to the goal cell if the box is in the same cell, or do nothing.",
        f"Grid size: {env.GRID_WIDTH} rows x {env.GRID_HEIGHT} columns\n",
        "Boxes:"
    ]

    for box in env.boxes:
        for i, pos in enumerate(box.positions):
            goal = env.goals[box.color][i]
            lines.append(f"- {box.color} box at {pos}, goal at {goal}")

    lines.append("\nAgents:")
    for i, agent in enumerate(env.agents):
        lines.append(f"- Agent {i} at {agent.position}")

    lines.append("\nFor example, if the blue box is at (0, 0) and the goal is at (1, 1),")
    lines.append("the agent can move the blue box from (0, 0) to (1, 0) and then to (1, 1).")
    lines.append("\nPlease return an ordered list of actions in the following format:")
    lines.append("- Agent [id]: move [color] box from (x, y) to (new x, new y) [direction]")
    

    return "\n".join(lines)

formatted_prompt = format_prompt(BoxNet1.BoxNet1())
print(formatted_prompt)
def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
        messages=[{"role": "system", "content": "You are a helpful robot task planner."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    total_tokens = response.usage.total_tokens
    print(f"Total tokens used: {total_tokens}")
    return response.choices[0].message.content

res = call_llm(format_prompt(BoxNet1.BoxNet1()))
print(res)