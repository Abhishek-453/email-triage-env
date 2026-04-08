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

# ── Env vars: use .get() with safe defaults so script never crashes on missing vars ──
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:8000")

# Warn early if critical vars are missing (don't crash — let validator see the log)
_missing = [v for v in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"] if not os.environ.get(v)]
if _missing:
    print(f"[WARNING] Missing env vars: {_missing}. Using defaults — scores may be 0.")

client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)


def llm(prompt: str, max_tokens: int = 150) -> str:
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


def wait_for_server(url: str, retries: int = 20, delay: float = 1.5):
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
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


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
            reward = resp.json()["reward"]
            total_reward += reward
            print(f"  [{task_name}] email={email['id']}  reward={reward:.1f}")

        except Exception as e:
            print(f"  [{task_name}] email={email['id']}  ERROR: {e}  reward=0.0")
            total_reward += 0.0

    score = total_reward / len(inbox) if inbox else 0.0
    return score


def main():
    start_time = time.time()
    print("=" * 60)
    print("email-triage-env  |  Inference Script")
    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
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
                # FIX: /reset is a POST endpoint, not GET
                reset_resp = requests.post(f"{ENV_URL}/reset", timeout=30)
                reset_resp.raise_for_status()
                inbox = reset_resp.json()["inbox"]
                print(f"   Inbox size: {len(inbox)} emails")
            except Exception as e:
                print(f"   [ERROR] Could not reset env: {e}")
                scores[task] = 0.0
                continue

            score = run_task(task, inbox)
            scores[task] = score
            print(f"   Score: {score:.4f}")

        overall = sum(scores.values()) / len(scores) if scores else 0.0
        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for task, s in scores.items():
            status = "PASS" if s >= 0.5 else "FAIL"
            print(f"  [{status}]  {task:<20}  {s:.4f}")
        print(f"\n  Overall Score : {overall:.4f}")
        print(f"  Elapsed Time  : {elapsed:.1f}s")
        print("=" * 60)

        for task, s in scores.items():
            assert 0.0 <= s <= 1.0, f"Score out of range for {task}: {s}"

        return scores

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        return {}

    finally:
        if server_proc:
            server_proc.terminate()


if __name__ == "__main__":
    main()