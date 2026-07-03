"""
In this file I am building the multi-agent slow path with LangGraph.
The classify and severity agents call a REAL local model (llama3.2 via Ollama).

Small local models are weak at arithmetic, so I do the comparison in Python and hand
the model conclusion-ready facts (recent average, how far below/above). The model then
only has to NAME the pattern, not compute it. Giving a weak model less to figure out
is the key to making it reliable.
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
import ollama

MODEL = "llama3.2"


def ask_llm(prompt: str) -> str:
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip().lower()


class AnomalyState(TypedDict):
    timestamp: str
    value: float
    recent: list
    anomaly_type: Optional[str]
    severity: Optional[str]
    alert_sent: Optional[bool]
    alert_message: Optional[str]


def context_agent(state: AnomalyState) -> dict:
    print(f"[context] anomaly at {state['timestamp']} value={state['value']:.2f}")
    return {"recent": state.get("recent", [])}


def classify_agent(state: AnomalyState) -> dict:
    recent = state.get("recent", [])
    avg = sum(recent) / len(recent) if recent else state["value"]
    diff = state["value"] - avg
    direction = "far below" if diff < -10 else "far above" if diff > 10 else "close to"
    prompt = (
        "A machine sensor reading is being classified.\n"
        f"The reading is {state['value']:.1f}. The recent average is {avg:.1f}. "
        f"So the reading is {direction} the recent average.\n"
        "Reply with exactly ONE word:\n"
        "- 'drop' if the reading is far below the average\n"
        "- 'spike' if the reading is far above the average\n"
        "- 'drift' if the reading is close to the average\n"
        "One word only."
    )
    answer = ask_llm(prompt)
    kind = next((k for k in ("spike", "drop", "drift") if k in answer), "drift")
    print(f"[classify] reading {state['value']:.1f} vs avg {avg:.1f} ({direction}); "
          f"LLM said '{answer}' -> {kind}")
    return {"anomaly_type": kind}


def severity_agent(state: AnomalyState) -> dict:
    distance = abs(state["value"] - 89)
    closeness = ("very far from" if distance > 40 else
                 "moderately far from" if distance > 20 else "near")
    prompt = (
        "A machine sensor anomaly needs a severity rating.\n"
        f"Normal is about 89. The reading is {state['value']:.1f}, which is "
        f"{closeness} normal. The anomaly type is {state['anomaly_type']}.\n"
        "Reply with exactly ONE word:\n"
        "- 'high' if the reading is very far from normal\n"
        "- 'medium' if moderately far\n"
        "- 'low' if near normal\n"
        "One word only."
    )
    answer = ask_llm(prompt)
    sev = next((s for s in ("high", "medium", "low") if s in answer), "medium")
    print(f"[severity] reading {state['value']:.1f} is {closeness} normal; "
          f"LLM said '{answer}' -> {sev}")
    return {"severity": sev}


def alert_agent(state: AnomalyState) -> dict:
    msg = (f"ALERT [{state['severity']}] {state['anomaly_type']} anomaly at "
           f"{state['timestamp']}, value {state['value']:.2f}")
    print(f"[alert] {msg}")
    return {"alert_sent": True, "alert_message": msg}


def route_after_severity(state: AnomalyState) -> str:
    if state["severity"] == "low":
        print("[route] low severity, logging only, no alert")
        return "log_only"
    return "alert"


def build_graph():
    g = StateGraph(AnomalyState)
    g.add_node("context", context_agent)
    g.add_node("classify", classify_agent)
    g.add_node("severity", severity_agent)
    g.add_node("alert", alert_agent)
    g.add_edge(START, "context")
    g.add_edge("context", "classify")
    g.add_edge("classify", "severity")
    g.add_conditional_edges("severity", route_after_severity,
                            {"alert": "alert", "log_only": END})
    g.add_edge("alert", END)
    return g.compile()


if __name__ == "__main__":
    graph = build_graph()

    print("=== test 1: severe drop (value 2.08) ===")
    graph.invoke({
        "timestamp": "2013-12-16 17:25:00", "value": 2.08,
        "recent": [90.0, 88.0, 70.0, 40.0, 10.0],
        "anomaly_type": None, "severity": None,
        "alert_sent": None, "alert_message": None,
    })

    print("\n=== test 2: mild wobble (value 82) ===")
    graph.invoke({
        "timestamp": "2014-01-01 12:00:00", "value": 82.0,
        "recent": [89.0, 90.0, 88.0, 91.0, 89.0],
        "anomaly_type": None, "severity": None,
        "alert_sent": None, "alert_message": None,
    })
