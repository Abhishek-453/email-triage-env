"""
validate.py — Pre-submission validation script
"""

import os
import sys
import time
import subprocess
import requests
import yaml

CHECKS = []
PASSED = []
FAILED = []


def check(name: str):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


@check("openenv.yaml valid")
def check_yaml():
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    assert "name" in data
    assert "tasks" in data and len(data["tasks"]) >= 3, "Need 3+ tasks"
    assert "endpoints" in data
    for key in ["reset", "step", "state"]:
        assert key in data["endpoints"], f"Missing endpoint: {key}"
    return f"{len(data['tasks'])} tasks defined"


@check("Required files present")
def check_files():
    required = ["inference.py", "Dockerfile", "requirements.txt", "app.py"]
    missing = [f for f in required if not os.path.isfile(f)]
    assert not missing, f"Missing files: {missing}"
    return "All required files found"


@check("Env vars defined")
def check_env_vars():
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        return f"WARNING: Not set locally (must be set in deployment): {missing}"
    return "All env vars set"


@check("Server starts and /reset returns 200")
def check_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8765"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        for _ in range(20):
            try:
                r = requests.get("http://127.0.0.1:8765/", timeout=3)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.8)

        # Health check
        r = requests.get("http://127.0.0.1:8765/", timeout=5)
        assert r.status_code == 200, f"/ returned {r.status_code}"

        # reset() pehle call karo
        r = requests.get("http://127.0.0.1:8765/reset", timeout=10)
        assert r.status_code == 200, f"/reset returned {r.status_code}"
        data = r.json()
        assert "inbox" in data, "reset() response missing 'inbox'"

        # reset ke BAAD state() call karo
        r = requests.get("http://127.0.0.1:8765/state", timeout=5)
        assert r.status_code == 200, f"/state returned {r.status_code}"

        # step() test karo
        inbox = data["inbox"]
        assert inbox, "Inbox is empty after reset"
        email_id = inbox[0]["id"]
        r = requests.post(
            "http://127.0.0.1:8765/step",
            json={"action": "classify", "email_id": email_id, "value": "HIGH"},
            timeout=5,
        )
        assert r.status_code == 200, f"/step returned {r.status_code}"
        step_data = r.json()
        assert "reward" in step_data
        assert 0.0 <= step_data["reward"] <= 1.0

        return "Server OK — /reset /step /state all pass"
    finally:
        proc.terminate()


@check("Task graders score in [0.0, 1.0]")
def check_graders():
    import importlib.util
    import unittest.mock as mock

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8766"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        for _ in range(20):
            try:
                r = requests.get("http://127.0.0.1:8766/", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.8)

        scores = {}
        graders = [
            ("tasks/priority_grader.py", "priority_classification"),
            ("tasks/spam_grader.py", "spam_detection"),
            ("tasks/reply_grader.py", "reply_drafting"),
        ]

        for grader_path, task_id in graders:
            spec = importlib.util.spec_from_file_location(task_id, grader_path)
            mod = importlib.util.module_from_spec(spec)

            os.environ.setdefault("API_BASE_URL", "http://stub")
            os.environ.setdefault("MODEL_NAME", "stub-model")
            os.environ.setdefault("HF_TOKEN", "stub-token")

            with mock.patch("openai.OpenAI") as mock_openai:
                instance = mock_openai.return_value
                instance.chat.completions.create.return_value = mock.MagicMock(
                    choices=[mock.MagicMock(message=mock.MagicMock(content="HIGH"))]
                )
                try:
                    spec.loader.exec_module(mod)
                    score = mod.grade(base_url="http://127.0.0.1:8766")
                    assert 0.0 <= score <= 1.0, f"{task_id} score={score} out of [0,1]"
                    scores[task_id] = score
                except Exception as e:
                    raise AssertionError(f"Grader {grader_path} failed: {e}")

        return f"Grader scores: {scores}"
    finally:
        proc.terminate()


def main():
    print("=" * 60)
    print("Pre-Submission Validator — email-triage-env")
    print("=" * 60)

    for name, fn in CHECKS:
        print(f"\n[CHECK] {name}")
        try:
            result = fn()
            print(f"  OK  {result}")
            PASSED.append(name)
        except Exception as e:
            print(f"  FAILED  {e}")
            FAILED.append(name)

    print("\n" + "=" * 60)
    print(f"PASSED: {len(PASSED)}/{len(CHECKS)}")
    if FAILED:
        print(f"FAILED: {FAILED}")
        print("\nFix the above before submitting.")
        sys.exit(1)
    else:
        print("All checks passed. Ready to submit!")


if __name__ == "__main__":
    main()
