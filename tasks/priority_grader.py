"""Priority Classification Grader — score in [0.0, 1.0]"""

import requests
import os


def grade(base_url: str = "http://localhost:8000") -> float:
    r = requests.get(f"{base_url}/reset", timeout=30)
    r.raise_for_status()
    inbox = r.json()["inbox"]

    PRIORITY_PROMPT = """You are an email triage assistant.
Given the email below, classify its priority as exactly one of: HIGH, MEDIUM, or LOW.
Respond with ONLY the single word: HIGH, MEDIUM, or LOW.

From: {from_}
Subject: {subject}
Body: {body}
"""

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["HF_TOKEN"],
        base_url=os.environ["API_BASE_URL"],
    )
    model = os.environ["MODEL_NAME"]

    correct = 0
    for email in inbox:
        prompt = PRIORITY_PROMPT.format(
            from_=email["from"],
            subject=email["subject"],
            body=email["body"],
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        predicted = completion.choices[0].message.content.strip().upper()

        resp = requests.post(
            f"{base_url}/step",
            json={"action": "classify", "email_id": email["id"], "value": predicted},
            timeout=30,
        )
        resp.raise_for_status()
        correct += resp.json()["reward"]

    score = correct / len(inbox) if inbox else 0.0
    assert 0.0 <= score <= 1.0
    print(f"[priority_grader] score={score:.2f}")
    return score


if __name__ == "__main__":
    grade()