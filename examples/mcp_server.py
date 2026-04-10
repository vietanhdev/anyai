"""MCP server exposing AI tools for Claude/GPT agents.

Usage: python mcp_tools.py
   or: anyai mcp-server

This starts a Model Context Protocol server that exposes OCR, detection,
summarization, sentiment analysis, and chat as tools for AI agents.

Install: pip install anyai[all]
"""
from anyai.mcp_server import create_server


def main():
    """Start the AnyAI MCP server."""
    server = create_server()

    # Show registered tools
    print("AnyAI MCP Server")
    print("=" * 40)
    print("Registered tools:")
    print("  - detect_objects    (anycv)")
    print("  - read_text         (anyocr)")
    print("  - chat              (anyllm)")
    print("  - summarize_text    (anynlp)")
    print("  - analyze_sentiment (anynlp)")
    print("  - profile_data      (tableai)")
    print("  - classify_data     (anyml)")
    print()
    print("Connect from Claude Desktop or any MCP client.")
    print("Config for claude_desktop_config.json:")
    print('  {"mcpServers": {"anyai": {"command": "anyai", "args": ["mcp-server"]}}}')
    print()

    # Start the server (stdio transport by default)
    server.run(transport="stdio")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install anyai[all] mcp")
