"""
In this file I test my MCP alert server the way a real MCP client would: I launch the
server, connect over stdio, list its tools, and call send_alert once. This proves the
server is a genuine, working MCP server.
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command="uv", args=["run", "python", "mcp_alert_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools exposed by the server:", [t.name for t in tools.tools])

            result = await session.call_tool("send_alert", {
                "timestamp": "2013-12-16 17:25:00",
                "value": 2.08,
                "anomaly_type": "drop",
                "severity": "high",
            })
            print("tool result:", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
