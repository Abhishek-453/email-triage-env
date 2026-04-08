"""
inference.py — Root-level inference script
Uses OpenAI-compatible client as required by submission rules.
"""

import os
import sys
import time
import subprocess
import requests
from openai import OpenAI

# ── Updated Ports to match Dockerfile (7860) ──
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

# OpenAI Client setup with robust error handling to prevent Phase 2 crashes
client = None
try:
    # Ensure base_url is clean and has /v1 suffix if missing for HF
    clean_url = API_BASE_URL.strip() if API_BASE_URL else "https://api-inference.huggingface.co/v1"
    if "huggingface.co" in clean_url and not clean_url.endswith("/v1"):
        clean_url = clean_url.rstrip("/") + "/v1"
    
    client = OpenAI(
        api_key=HF_TOKEN if HF_TOKEN else "dummy_token",
        base_url=clean_url
    )
    print(f"[DEBUG] Client initialized for URL: {clean_url}")
except Exception as e:
    print(f"[FATAL] Client failed to initialize: {e}")

def llm(prompt: str, max_tokens: int = 150) -> str:
    if client is None:
        return ""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[llm ERROR] {e}")
        return ""

def wait_for_server(url: str, retries: int = 25, delay: float = 2.0):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"[inference] Server is up at {url}")
                return
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError(f"Server at {url} did not start in time.")

def start_server():
    # Using Port 7860 to match Dockerfile EXPOSE and HEALTHCHECK 
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

def run_task(task_name: str, inbox: list) -> float:
    prompts = {
        "classify": (
            "Classify this email priority as HIGH, MEDIUM, or LOW.\n"
            "Reply with ONE word only — exactly HIGH, MEDIUM, or LOW.\n\n"
            "From: {from_}\nSubject: {subject}\nBody: {body}"
        ),
        "spam_check": (
            "Is this email spam? Reply with ONLY the word true or false.\n\n"
            "From: {from_}\nSubject: {subject}\nBody: {body}"
        ),
        "reply": (
            "Draft a short professional reply (20-80 words) to this email.\n\n"
            "From: {from_}\nSubject: {subject}\nBody: {body}\n\nReply:"
        ),
    }

    total_reward = 0.0
    for email in inbox:
        try:
            prompt = prompts[task_name].format(
                from_=email["from"],
                subject=email["subject"],
                body=email["body"],
            )
            value = llm(prompt, max_tokens=150 if task_name == "reply" else 10)

            if task_name == "spam_check":
                value = value.lower().startswith("true")

            resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": task_name, "email_id": email["id"], "value": value},
                timeout=30,
            )
            resp.raise_for_status()
            reward = resp.json().get("reward", 0.0)
            total_reward += reward
        except Exception as e:
            print(f"  [{task_name}] email={email['id']} ERROR: {e}")
            total_reward += 0.0

    return total_reward / len(inbox) if inbox else 0.0

def main():
    start_time = time.time()
    print("=" * 60)
    print("email-triage-env  |  Inference Script")
    print("=" * 60)

    server_proc = None
    try:
        server_proc = start_server()
        wait_for_server(f"{ENV_URL}/")

        scores = {}
        tasks = ["classify", "spam_check", "reply"]

        for task in tasks:
            print(f"\n-- Task: {task} --")
            try:
                # /reset is a POST endpoint in app.py [cite: 1]
                reset_resp = requests.post(f"{ENV_URL}/reset", timeout=30)
                reset_resp.raise_for_status()
                inbox = reset_resp.json()["inbox"]
                print(f"   Inbox size: {len(inbox)} emails")
                
                score = run_task(task, inbox)
                scores[task] = score
                print(f"   Score: {score:.4f}")
            except Exception as e:
                print(f"   [ERROR] Task {task} failed: {e}")
                scores[task] = 0.0

        overall = sum(scores.values()) / len(scores) if scores else 0.0
        print("\n" + "=" * 60)
        print(f"RESULTS | Overall Score : {overall:.4f}")
        print("=" * 60)

        return scores

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        return {}

    finally:
        if server_proc:
            server_proc.terminate()

if __name__ == "__main__":
    main()