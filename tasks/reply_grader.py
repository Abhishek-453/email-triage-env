"""Reply Drafting Grader — score in [0.0, 1.0]"""

import requests
import os


def grade(base_url: str = "http://localhost:8000") -> float:
    r = requests.get(f"{base_url}/reset", timeout=30)
    r.raise_for_status()
    inbox = r.json()["inbox"]

    REPLY_PROMPT = """You are a professional email assistant.
Draft a concise, polite reply to the email below. Keep it under 80 words.

From: {from_}
Subject: {subject}
Body: {body}

Reply:"""

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["HF_TOKEN"],
        base_url=os.environ["API_BASE_URL"],
    )
    model = os.environ["MODEL_NAME"]

    total_reward = 0.0
    for email in inbox:
        prompt = REPLY_PROMPT.format(
            from_=email["from"],
            subject=email["subject"],
            body=email["body"],
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
        )
        reply_text = completion.choices[0].message.content.strip()

        resp = requests.post(
            f"{base_url}/step",
            json={"action": "reply", "email_id": email["id"], "value": reply_text},
            timeout=30,
        )
        resp.raise_for_status()
        total_reward += resp.json()["reward"]

    score = total_reward / len(inbox) if inbox else 0.0
    assert 0.0 <= score <= 1.0
    print(f"[reply_grader] score={score:.2f}")
    return score


if __name__ == "__main__":
    grade()