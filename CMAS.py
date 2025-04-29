import BoxNet1
import BoxNet2
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def format_prompt(env):
    """Format prompt for centralized CMAS planner."""
    lines = [
        "You are a centralized task planner for a multi-robot system."
    ]

    if isinstance(env, BoxNet1.BoxNet1):
        lines.append(f"Grid size: {env.GRID_WIDTH} rows x {env.GRID_HEIGHT} columns (BoxNet1)")

        lines.append("\nBoxes:")
        for box in env.boxes:
            for pos in box.positions:
                lines.append(f"- {box.color} box at {pos}, goal at {env.goals[box.color][0]}")

        lines.append("\nAgents:")
        for i, agent in enumerate(env.agents):
            lines.append(f"- Agent {i} at {agent.position}")

    elif isinstance(env, BoxNet2.BoxNet2):
        lines.append(f"Environment: BoxNet2 using corners.")

        lines.append("\nBoxes:")
        for box in env.boxes:
            lines.append(f"- {box.color} box at corner {box.position}, goal at {env.goals[box.color][0]}")

        lines.append("\nAgents:")
        for i, agent in enumerate(env.agents):
            lines.append(f"- Agent {i} at {agent.cell_position}")

    lines.append("\nFormat your plan as a list:")
    lines.append("- Agent [id]: move [color] box from (x, y) to (x, y) [direction]")
    lines.append("- Agent [id]: do nothing")

    return "\n".join(lines)

def call_llm(prompt):
    """Send the centralized prompt to the LLM."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a centralized multi-robot system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response
