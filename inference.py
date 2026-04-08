"""
inference.py — Root-level inference script with Structured Logging
"""

import os
import sys
import time
import subprocess
import requests
from openai import OpenAI

# ── Configuration ──
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

# OpenAI Client Setup
client = None
try:
    clean_url = API_BASE_URL.strip() if API_BASE_URL else "https://api-inference.huggingface.co/v1"
    if "huggingface.co" in clean_url and not clean_url.endswith("/v1"):
        clean_url = clean_url.rstrip("/") + "/v1"
    
    client = OpenAI(api_key=HF_TOKEN if HF_TOKEN else "dummy", base_url=clean_url)
except Exception as e:
    print(f"Init Error: {e}", flush=True)

def llm(prompt: str, max_tokens: int = 150) -> str:
    if not client: return ""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return ""

def wait_for_server(url: str):
    for _ in range(30):
        try:
            if requests.get(url, timeout=5).status_code == 200: return
        except: pass
        time.sleep(2)

def run_task(task_name: str, inbox: list):
    # --- [START] REQUIRED BLOCK ---
    print(f"[START] task={task_name}", flush=True)
    
    prompts = {
        "classify": "Classify priority: HIGH, MEDIUM, or LOW. Reply with ONE word only.\n\nEmail: {body}",
        "spam_check": "Is this spam? Reply ONLY true or false.\n\nEmail: {body}",
        "reply": "Draft a short professional reply.\n\nEmail: {body}"
    }

    total_reward = 0.0
    for i, email in enumerate(inbox):
        try:
            prompt = prompts[task_name].format(body=email["body"])
            value = llm(prompt, max_tokens=100 if task_name == "reply" else 5)
            
            if task_name == "spam_check":
                value = value.lower().startswith("true")

            resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": task_name, "email_id": email["id"], "value": value},
                timeout=30
            )
            reward = resp.json().get("reward", 0.0)
            total_reward += reward
            
            # --- [STEP] REQUIRED BLOCK ---
            print(f"[STEP] step={i+1} reward={reward:.2f}", flush=True)

        except Exception:
            print(f"[STEP] step={i+1} reward=0.0", flush=True)

    score = total_reward / len(inbox) if inbox else 0.0
    # --- [END] REQUIRED BLOCK ---
    print(f"[END] task={task_name} score={score:.4f} steps={len(inbox)}", flush=True)
    return score

def main():
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    try:
        wait_for_server(f"{ENV_URL}/")
        tasks = ["classify", "spam_check", "reply"]
        for task in tasks:
            reset_resp = requests.post(f"{ENV_URL}/reset", timeout=30)
            inbox = reset_resp.json().get("inbox", [])
            run_task(task, inbox)
    finally:
        server_proc.terminate()

if __name__ == "__main__":
    main()