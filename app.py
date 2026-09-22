"""A local review queue; session data is cleared when the session ends."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from intentlab.service import RoutingRequest, RoutingService

st.set_page_config(page_title="Banking request routing", layout="wide")
st.title("Banking request routing")
st.caption("77 support intents · calibrated confidence · human review for uncertain requests")
report = Path("reports/metrics.json")
if report.exists():
    metrics = json.loads(report.read_text())
    point = metrics["routing_test"]
    a, b, c = st.columns(3)
    a.metric(
        "Overall test accuracy",
        f"{metrics['test'][metrics['selected_model']]['accuracy']:.1%}",
    )
    b.metric("Automatically routed", f"{point['coverage']:.1%}")
    c.metric(
        "Accuracy among routed",
        f"{point['accuracy']:.1%}" if point["accuracy"] is not None else "N/A",
    )
try:
    service = st.cache_resource(RoutingService)()
except (OSError, ValueError):
    st.info("Train the model first: python -m intentlab.train")
    st.stop()
st.session_state.setdefault("queue", [])
with st.form("request"):
    text = st.text_area("Customer request", "My card has not arrived yet.", max_chars=2000)
    submitted = st.form_submit_button("Route request")
if submitted:
    try:
        result = service.predict(RoutingRequest(text=text))
        if result["action"] == "human_review":
            st.warning("Needs human review")
            st.session_state.queue.append(
                {
                    "text": text,
                    "suggested_intent": result["candidates"][0]["intent"],
                    "confidence": result["confidence"],
                    "model_version": result["model_version"],
                }
            )
        else:
            st.success(f"Route to: {result['intent']}")
        st.dataframe(pd.DataFrame(result["candidates"]), hide_index=True)
        st.caption(f"Model {result['model_version']} · review threshold {result['threshold']:.2f}")
    except ValueError as exc:
        st.error(str(exc))
st.subheader("Human review queue")
st.caption(
    "Session-only demonstration. Use sample text without account numbers or personal information."
)
st.dataframe(pd.DataFrame(st.session_state.queue), hide_index=True)
if st.button("Clear review queue"):
    st.session_state.queue = []
    st.rerun()
st.info(
    "High confidence does not prove a request is in scope. Unsupported requests remain a known failure mode; see the evaluation report."
)
if report.exists():
    with st.expander("Routing trade-off"):
        st.image("reports/routing.svg")
