"""
email-triage-env — OpenEnv-compliant FastAPI server
Endpoints: GET /reset, POST /step, GET /state
"""

import os
import json
import random
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")

EMAILS_CORPUS = [
    {
        "id": "e001",
        "from": "ceo@company.com",
        "subject": "Urgent: Board meeting rescheduled to tomorrow 9AM",
        "body": "Please confirm attendance immediately. All department heads must be present.",
        "true_priority": "HIGH",
        "true_spam": False,
    },
    {
        "id": "e002",
        "from": "newsletter@deals123.biz",
        "subject": "YOU WON $1,000,000!!! Click here NOW!!!",
        "body": "Congratulations! You have been selected. Send your bank details to claim prize.",
        "true_priority": "LOW",
        "true_spam": True,
    },
    {
        "id": "e003",
        "from": "hr@company.com",
        "subject": "Reminder: Submit timesheets by Friday",
        "body": "Please submit your timesheets before end of day Friday. Thank you.",
        "true_priority": "MEDIUM",
        "true_spam": False,
    },
    {
        "id": "e004",
        "from": "client@bigcorp.com",
        "subject": "Contract renewal — need response ASAP",
        "body": "Our contract expires in 48 hours. Please confirm renewal terms immediately.",
        "true_priority": "HIGH",
        "true_spam": False,
    },
    {
        "id": "e005",
        "from": "noreply@pharmacy-cheap.ru",
        "subject": "Buy meds cheap no prescription",
        "body": "Get all medicines without prescription. Lowest prices guaranteed!",
        "true_priority": "LOW",
        "true_spam": True,
    },
    {
        "id": "e006",
        "from": "devops@company.com",
        "subject": "Scheduled maintenance Sunday 2AM-4AM",
        "body": "Systems will be down for maintenance. No action required from your side.",
        "true_priority": "MEDIUM",
        "true_spam": False,
    },
]

_state: Dict[str, Any] = {}


def _fresh_state() -> Dict[str, Any]:
    emails = random.sample(EMAILS_CORPUS, k=min(4, len(EMAILS_CORPUS)))
    return {
        "inbox": [
            {
                "id": e["id"],
                "from": e["from"],
                "subject": e["subject"],
                "body": e["body"],
            }
            for e in emails
        ],
        "_ground_truth": {e["id"]: e for e in emails},
        "agent_actions": [],
        "done": False,
        "step_count": 0,
    }


class StepAction(BaseModel):
    action: str
    email_id: str
    value: Any


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    inbox: List[Dict]
    agent_actions: List[Dict]
    done: bool
    step_count: int


class ResetResponse(BaseModel):
    status: str
    inbox: List[Dict]
    message: str


@app.get("/", status_code=200)
def health():
    return {"status": "ok", "env": "email-triage-env"}


@app.post("/reset", response_model=ResetResponse)
def reset():
    global _state
    _state = _fresh_state()
    return ResetResponse(
        status="ok",
        inbox=_state["inbox"],
        message=f"Inbox loaded with {len(_state['inbox'])} emails. Use POST /step to act.",
    )


@app.post("/step", response_model=StepResponse)
def step(action: StepAction):
    global _state
    if not _state:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call GET /reset first.")

    gt = _state["_ground_truth"]
    if action.email_id not in gt:
        raise HTTPException(status_code=404, detail=f"Email id '{action.email_id}' not in current inbox.")

    email = gt[action.email_id]
    reward = 0.0
    observation = ""

    if action.action == "classify":
        predicted = str(action.value).upper()
        correct = email["true_priority"]
        reward = 1.0 if predicted == correct else 0.0
        observation = f"Classification for {action.email_id}: predicted={predicted}, correct={correct}"

    elif action.action == "spam_check":
        predicted = bool(action.value)
        correct = email["true_spam"]
        reward = 1.0 if predicted == correct else 0.0
        observation = f"Spam check for {action.email_id}: predicted={predicted}, correct={correct}"

    elif action.action == "reply":
        reply_text = str(action.value).strip()
        if email["true_priority"] == "HIGH" and len(reply_text) > 20:
            reward = 1.0
        elif len(reply_text) > 10:
            reward = 0.5
        else:
            reward = 0.0
        observation = f"Reply drafted for {action.email_id} (length={len(reply_text)}). Reward={reward}"

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action '{action.action}'.")
    
    _state["agent_actions"].append(
        {"action": action.action, "email_id": action.email_id, "value": action.value, "reward": reward}
    )
    _state["step_count"] += 1

    total_possible = len(_state["inbox"]) * 3
    _state["done"] = _state["step_count"] >= total_possible

    return StepResponse(
        observation=observation,
        reward=reward,
        done=_state["done"],
        info={"step_count": _state["step_count"]},
    )


@app.get("/state", response_model=StateResponse)
def state():
    if not _state:
        reset()
    return StateResponse(
        inbox=_state["inbox"],
        agent_actions=_state["agent_actions"],
        done=_state["done"],
        step_count=_state["step_count"],
    )