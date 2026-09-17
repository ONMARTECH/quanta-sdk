"""Export Quanta MCP server tools and register with Antigravity."""

import asyncio
import json
from pathlib import Path

from quanta.mcp_server import mcp

ANTIGRAVITY_MCP_DIR = Path("/Users/aes/.gemini/antigravity/mcp/quanta")
ANTIGRAVITY_CONFIG_FILE = Path("/Users/aes/.gemini/antigravity/mcp_config.json")
VENV_PYTHON = "/Users/aes/Antigravity Projects/Alfa/quanta/.venv/bin/python"
QUANTA_DIR = "/Users/aes/Antigravity Projects/Alfa/quanta"


async def export_tools() -> None:
    ANTIGRAVITY_MCP_DIR.mkdir(parents=True, exist_ok=True)

    tools = await mcp.list_tools()
    print(f"Exporting {len(tools)} Quanta tools to {ANTIGRAVITY_MCP_DIR}...")

    for tool in tools:
        tool_data = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters or {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
        tool_file = ANTIGRAVITY_MCP_DIR / f"{tool.name}.json"
        with open(tool_file, "w", encoding="utf-8") as f:
            json.dump(tool_data, f, indent=2)
        print(f"  ✓ {tool.name}.json")

    # Write instructions.md
    instructions_content = (
        "Quanta Quantum SDK MCP Server: Provides 23 AI-native quantum tools including "
        "circuit execution (`run_circuit`), reasoning evaluation (`quanta_reasoning_eval`), "
        "fault-tolerant cost estimation (`estimate_fault_tolerant_cost`), QEC simulation, "
        "and Apple Silicon Metal GPU acceleration.\n"
    )
    with open(ANTIGRAVITY_MCP_DIR / "instructions.md", "w", encoding="utf-8") as f:
        f.write(instructions_content)
    print("  ✓ instructions.md written.")

    # Register in mcp_config.json if not already present
    if ANTIGRAVITY_CONFIG_FILE.exists():
        with open(ANTIGRAVITY_CONFIG_FILE, encoding="utf-8") as f:
            config = json.load(f)

        mcp_servers = config.setdefault("mcpServers", {})
        mcp_servers["quanta"] = {
            "command": VENV_PYTHON,
            "args": ["-m", "quanta.mcp_server"],
            "env": {
                "PYTHONPATH": QUANTA_DIR,
            },
        }

        with open(ANTIGRAVITY_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Successfully registered 'quanta' in {ANTIGRAVITY_CONFIG_FILE}")


if __name__ == "__main__":
    asyncio.run(export_tools())
