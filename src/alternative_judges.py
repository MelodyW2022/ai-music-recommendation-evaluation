from copy import deepcopy
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .guardrails import GuardrailViolation
from .llm_explainer import (
    JUDGE_INSTRUCTIONS,
    JUDGE_RESPONSE_SCHEMA,
    build_judge_payload,
    parse_judge_violations,
)
from .retriever import RetrievalResult


CLAUDE_JUDGE_MODEL = "claude-haiku-5-5"
GROQ_JUDGE_MODEL = "openai/gpt-oss-20b"

STRICT_JUDGE_SCHEMA = deepcopy(JUDGE_RESPONSE_SCHEMA)
STRICT_JUDGE_SCHEMA["additionalProperties"] = False
STRICT_JUDGE_SCHEMA["properties"]["claims"]["items"]["additionalProperties"] = False
del STRICT_JUDGE_SCHEMA["properties"]["claims"]["minItems"]


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    req = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urlopen(req, timeout=45) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"{req.host} returned HTTP {exc.code}") from None


def review_with_claude(
    explanation: str, result: RetrievalResult
) -> list[GuardrailViolation]:
    response = _post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
        {
            "model": CLAUDE_JUDGE_MODEL,
            "max_tokens": 2048,
            "system": JUDGE_INSTRUCTIONS,
            "messages": [
                {"role": "user", "content": json.dumps(build_judge_payload(explanation, result))}
            ],
            "output_config": {
                "format": {"type": "json_schema", "schema": STRICT_JUDGE_SCHEMA},
            },
        },
    )
    if response["stop_reason"] != "end_turn":
        raise ValueError("Claude did not complete its grounding review")
    text = "".join(block["text"] for block in response["content"] if block["type"] == "text")
    return parse_judge_violations(text)


def review_with_groq(
    explanation: str, result: RetrievalResult
) -> list[GuardrailViolation]:
    response = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        {
            "model": GROQ_JUDGE_MODEL,
            "max_completion_tokens": 2048,
            "reasoning_effort": "low",
            "messages": [
                {"role": "system", "content": JUDGE_INSTRUCTIONS},
                {"role": "user", "content": json.dumps(build_judge_payload(explanation, result))},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "grounding_review",
                    "strict": True,
                    "schema": STRICT_JUDGE_SCHEMA,
                },
            },
        },
    )
    choice = response["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("Groq did not complete its grounding review")
    return parse_judge_violations(choice["message"]["content"] or "")
