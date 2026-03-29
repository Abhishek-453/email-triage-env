# email-triage-env

An **OpenEnv-compliant** environment for the Scaler x Meta PyTorch OpenEnv Hackathon.

An AI agent must triage a simulated email inbox by classifying priority, detecting spam, and drafting replies.

---

## Tasks

| Task | Endpoint action | Score range |
|---|---|---|
| Priority Classification | `classify` | 0.0 – 1.0 |
| Spam Detection | `spam_check` | 0.0 – 1.0 |
| Reply Drafting | `reply` | 0.0 – 1.0 |

---

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_xxxx"

# 3. Start the environment server
uvicorn app:app --host 0.0.0.0 --port 8000

# 4. Run pre-submission validation
python validate.py

# 5. Run inference
python inference.py
```

---

## API Reference

### `GET /reset`
Resets environment, loads fresh inbox. Returns `{status, inbox, message}`.

### `POST /step`
```json
{
  "action": "classify" | "spam_check" | "reply",
  "email_id": "e001",
  "value": "HIGH" | true | "Dear Sir, ..."
}
```
Returns `{observation, reward, done, info}`.

### `GET /state`
Returns current `{inbox, agent_actions, done, step_count}`.

### `GET /`
Health check — returns `{status: "ok"}`.

---

## Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | OpenAI-compatible LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face / API key |