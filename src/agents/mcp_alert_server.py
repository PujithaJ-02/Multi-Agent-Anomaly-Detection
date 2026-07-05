"""
In this file I build a real MCP (Model Context Protocol) server that exposes a
"send_alert" tool. My alert agent can call this tool to dispatch a real alert when a
serious anomaly is detected, instead of just printing.

Why MCP: it is the standard way to give an agent access to external tools. Here the tool
writes a structured alert to a file (a real, honest destination; a Slack webhook or a
ticketing system could replace the file later without changing the agent, because MCP
keeps the tool interface the same).

This is a genuine MCP server built with FastMCP. Run it as its own process; a client
(my agent) connects and calls the send_alert tool.
"""
import json
import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# The alerts file lives at the project root so it is easy to find and demo.
ALERTS_FILE = Path(__file__).resolve().parents[2] / "alerts.log"

mcp = FastMCP("anomaly-alerts")


@mcp.tool()
def send_alert(timestamp: str, value: float, anomaly_type: str, severity: str) -> str:
    """Dispatch an anomaly alert. Writes a structured alert record to the alert log
    and returns a confirmation string.

    Args:
        timestamp: when the anomalous reading occurred
        value: the anomalous reading value
        anomaly_type: spike, drop, or drift
        severity: low, medium, or high
    """
    record = {
        "dispatched_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "timestamp": timestamp,
        "value": value,
        "anomaly_type": anomaly_type,
        "severity": severity,
    }
    with open(ALERTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return f"alert dispatched: {severity} {anomaly_type} at {timestamp} (value {value})"


if __name__ == "__main__":
    # Run the server over stdio (the standard local MCP transport).
    mcp.run()
