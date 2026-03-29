---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# email-triage-env

An **OpenEnv-compliant** environment for the Scaler x Meta PyTorch OpenEnv Hackathon.

An AI agent must triage a simulated email inbox by classifying priority, detecting spam, and drafting replies.

## Tasks

| Task | Endpoint action | Score range |
|---|---|---|
| Priority Classification | `classify` | 0.0 – 1.0 |
| Spam Detection | `spam_check` | 0.0 – 1.0 |
| Reply Drafting | `reply` | 0.0 – 1.0 |

## API Reference

GET /reset — Resets environment, loads fresh inbox.

POST /step — Agent performs an action on an email.

GET /state — Returns current inbox and agent actions.

GET / — Health check, returns status ok.

## Environment Variables

API_BASE_URL — OpenAI-compatible LLM API endpoint

MODEL_NAME — Model identifier

HF_TOKEN — Hugging Face API key