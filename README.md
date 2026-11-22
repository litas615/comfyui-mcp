# ComfyUI & A1111 MCP Server

A lightweight [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server built with `FastMCP` that bridges AI coding agents (Claude Code, Antigravity, Cursor, Codex) to local **ComfyUI** and **Stable Diffusion WebUI (A1111)** instances.

## Features

- **`get_comfyui_status`**: Health check and GPU VRAM inspection for local ComfyUI.
- **`render_art_with_comfyui`**: Queue workflow prompts, poll execution status, and retrieve generated image paths.
- **`get_a1111_status`**: Check local Stable Diffusion WebUI availability and loaded checkpoints.
- **`render_art_with_a1111`**: Trigger txt2img generation via A1111 API with configurable aspect ratios (`16:9`, `9:16`, `1:1`, etc.).

## Quick Start

```bash
pip install -r requirements.txt
export COMFYUI_URL=http://127.0.0.1:8188
export COMFYUI_OUTPUT_DIR=./output
python comfyui_mcp_server.py
```

## MCP Client Configuration

```json
{
  "mcpServers": {
    "comfyui-art": {
      "command": "python",
      "args": ["/path/to/comfyui-mcp/comfyui_mcp_server.py"]
    }
  }
}
```
