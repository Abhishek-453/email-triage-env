from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import random

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")

EMAILS_CORPUS = [
    {"id": "e001", "from": "ceo@company.com", "subject": "Urgent: Board meeting rescheduled to tomorrow 9AM", "body": "Please confirm attendance immediately.", "true_priority": "HIGH", "true_spam": False},
    {"id": "e002", "from": "newsletter@deals123.biz", "subject": "YOU WON $1,000,000!!!", "body": "Send your bank details to claim prize.", "true_priority": "LOW", "true_spam": True},
    {"id": "e003", "from": "hr@company.com", "subject": "Reminder: Submit timesheets by Friday", "body": "Please submit your timesheets before end of day Friday.", "true_priority": "MEDIUM", "true_spam": False},
    {"id": "e004", "from": "client@bigcorp.com", "subject": "Contract renewal — need response ASAP", "body": "Our contract expires in 48 hours. Please confirm renewal terms.", "true_priority": "HIGH", "true_spam": False},
    {"id": "e005", "from": "noreply@pharmacy-cheap.ru", "subject": "Buy meds cheap no prescription", "body": "Get all medicines without prescription.", "true_priority": "LOW", "true_spam": True},
    {"id": "e006", "from": "devops@company.com", "subject": "Scheduled maintenance Sunday 2AM-4AM", "body": "Systems will be down for maintenance.", "true_priority": "MEDIUM", "true_spam": False},
]

_state: Dict[str, Any] = {}

def _fresh_state():
    emails = random.sample(EMAILS_CORPUS, k=4)
    return {
        "inbox": [{"id": e["id"], "from": e["from"], "subject": e["subject"], "body": e["body"]} for e in emails],
        "_ground_truth": {e["id"]: e for e in emails},
        "agent_actions": [],
        "done": False,
        "step_count": 0,
    }

class StepAction(BaseModel):
    action: str
    email_id: str
    value: Any

@app.get("/")
def health():
    return {"status": "ok", "env": "email-triage-env"}

@app.post("/reset")
def reset():
    global _state
    _state = _fresh_state()
    return {"status": "ok", "inbox": _state["inbox"], "message": f"Inbox loaded with {len(_state['inbox'])} emails."}

@app.post("/step")
def step(action: StepAction):
    global _state
    if not _state:
        _state = _fresh_state()
    gt = _state["_ground_truth"]
    if action.email_id not in gt:
        raise HTTPException(status_code=404, detail=f"Email id '{action.email_id}' not found.")
    email = gt[action.email_id]
    reward = 0.0
    if action.action == "classify":
        reward = 1.0 if str(action.value).upper() == email["true_priority"] else 0.0
    elif action.action == "spam_check":
        reward = 1.0 if bool(action.value) == email["true_spam"] else 0.0
    elif action.action == "reply":
        reply_text = str(action.value).strip()
        reward = 1.0 if len(reply_text) > 20 else 0.5 if len(reply_text) > 10 else 0.0
    _state["agent_actions"].append({"action": action.action, "email_id": action.email_id, "reward": reward})
    _state["step_count"] += 1
    _state["done"] = _state["step_count"] >= len(_state["inbox"]) * 3
    return {"observation": f"Action {action.action} on {action.email_id}", "reward": reward, "done": _state["done"], "info": {"step_count": _state["step_count"]}}

@app.get("/state")
def state():
    global _state
    if not _state:
        _state = _fresh_state()
    return {"inbox": _state["inbox"], "agent_actions": _state["agent_actions"], "done": _state["done"], "step_count": _state["step_count"]}