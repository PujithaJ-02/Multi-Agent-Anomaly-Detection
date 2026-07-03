"""
In this file I am building the multi-agent slow path with LangGraph.
The classify and severity agents call a local model (llama3.2 via Ollama).

Classify compares against the machine's TRUE normal (~89), not the recent average.
If I used the recent average, a long sustained failure would keep looking "close to"
its own low readings and get wrongly called "drift" forever. Honest note: 89 is the
baseline from the training data, hardcoded here for the demo; a real system would
compute it from a long clean period.
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
import ollama

MODEL = "llama3.2"
NORMAL = 89.0


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
    diff = state["value"] - NORMAL
    direction = "far below" if diff < -10 else "far above" if diff > 10 else "close to"
    prompt = (
        "A machine sensor reading is being classified. "
        f"The reading is {state['value']:.1f}. Normal is about {NORMAL:.0f}. "
        f"So the reading is {direction} normal. "
        "Reply with exactly ONE word: "
        "'drop' if far below normal, 'spike' if far above normal, "
        "'drift' if close to normal. One word only."
    )
    answer = ask_llm(prompt)
    kind = next((k for k in ("spike", "drop", "drift") if k in answer), "drift")
    print(f"[classify] reading {state['value']:.1f} vs normal {NORMAL:.0f} ({direction}); "
          f"LLM said '{answer}' -> {kind}")
    return {"anomaly_type": kind}


def severity_agent(state: AnomalyState) -> dict:
    distance = abs(state["value"] - NORMAL)
    closeness = ("very far from" if distance > 40 else
                 "moderately far from" if distance > 20 else "near")
    prompt = (
        "A machine sensor anomaly needs a severity rating. "
        f"Normal is about {NORMAL:.0f}. The reading is {state['value']:.1f}, which is "
        f"{closeness} normal. The anomaly type is {state['anomaly_type']}. "
        "Reply with exactly ONE word: 'high' if very far from normal, "
        "'medium' if moderately far, 'low' if near. One word only."
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

    print("\n=== test 3: sustained low failure (value 33) ===")
    graph.invoke({
        "timestamp": "2014-02-08 22:00:00", "value": 33.0,
        "recent": [34.0, 33.0, 35.0, 32.0, 34.0],
        "anomaly_type": None, "severity": None,
        "alert_sent": None, "alert_message": None,
    })
