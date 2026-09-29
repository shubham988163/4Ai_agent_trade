"""Shared LLM wrapper for all agents — supports Anthropic (Claude) and Google (Gemini).

Provider selection (first match wins):
  1. LLM_PROVIDER env var ("gemini" or "anthropic"), if set
  2. GEMINI_API_KEY / GOOGLE_API_KEY set  -> Gemini
  3. otherwise                            -> Anthropic (ANTHROPIC_API_KEY / ant profile)

Rules enforced here regardless of provider (per the system design):
  - Structured calls are validated against a Pydantic schema — validated JSON
    or nothing.
  - Every call — prompt and response — is logged to the ledger's agent_log
    table (the audit trail).
  - Any failure (network, refusal, invalid output) returns the caller's
    fallback. Agents never raise into the trading path.
"""
from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

from trading.config import ANTHROPIC_MODEL, GEMINI_MODEL, GEMINI_FALLBACK_MODEL, AGENT_MAX_TOKENS
from trading.ledger import Ledger


T = TypeVar("T", bound=BaseModel)

_anthropic_client = None
_gemini_client = None


def provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").lower()
    if explicit in ("gemini", "anthropic"):
        return explicit
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic"


def credentials_available() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    return _gemini_client


# --- structured calls ---

def call_structured(agent_name: str, system: str, prompt: str,
                    output_model: type[T], fallback: T,
                    ledger: Ledger | None = None) -> T:
    """One-shot structured call. Returns a validated instance of output_model,
    or `fallback` on any failure."""
    ledger = ledger or Ledger()
    which = provider()
    model = GEMINI_MODEL if which == "gemini" else ANTHROPIC_MODEL
    try:
        if which == "gemini":
            result = _call_gemini_structured(system, prompt, output_model)
        else:
            result = _call_anthropic_structured(system, prompt, output_model)
        if result is None:
            ledger.log_agent_call(agent_name, model, prompt, None,
                                  ok=False, error="no parsed output (refusal/empty)")
            return fallback
        ledger.log_agent_call(agent_name, model, prompt,
                              result.model_dump_json(), ok=True)
        return result
    except Exception as e:  # noqa: BLE001 — agents must never crash the pipeline
        ledger.log_agent_call(agent_name, model, prompt, None,
                              ok=False, error=repr(e))
        return fallback


def _call_anthropic_structured(system: str, prompt: str, output_model: type[T]) -> T | None:
    response = _anthropic().messages.parse(
        model=ANTHROPIC_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=output_model,
    )
    if response.stop_reason == "refusal":
        return None
    return response.parsed_output


def _call_gemini_structured(system: str, prompt: str, output_model: type[T]) -> T | None:
    from google.genai import types
    last_err = None
    for model_name in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL):
        try:
            response = _gemini().models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=output_model,
                    max_output_tokens=AGENT_MAX_TOKENS,
                ),
            )
            parsed = response.parsed
            if parsed is None and response.text:
                # SDK didn't auto-parse; validate the JSON text ourselves.
                parsed = output_model.model_validate_json(response.text)
            if parsed is not None:
                return parsed
        except Exception as err:
            last_err = err
            continue
    if last_err is not None:
        raise last_err
    return None


# --- free-form text calls (EOD journal) ---

def call_text(agent_name: str, system: str, prompt: str,
              ledger: Ledger | None = None) -> str | None:
    """Free-form text call. Returns None on failure."""
    ledger = ledger or Ledger()
    which = provider()
    model = GEMINI_MODEL if which == "gemini" else ANTHROPIC_MODEL
    try:
        if which == "gemini":
            text = _call_gemini_text(system, prompt)
        else:
            text = _call_anthropic_text(system, prompt)
        if not text:
            ledger.log_agent_call(agent_name, model, prompt, None,
                                  ok=False, error="empty response/refusal")
            return None
        ledger.log_agent_call(agent_name, model, prompt, text, ok=True)
        return text
    except Exception as e:  # noqa: BLE001
        ledger.log_agent_call(agent_name, model, prompt, None,
                              ok=False, error=repr(e))
        return None


def _call_anthropic_text(system: str, prompt: str) -> str | None:
    # Streams to avoid HTTP timeouts on long reports.
    with _anthropic().messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        return None
    return "".join(b.text for b in message.content if b.type == "text")


def _call_gemini_text(system: str, prompt: str) -> str | None:
    from google.genai import types
    last_err = None
    for model_name in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL):
        try:
            response = _gemini().models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=AGENT_MAX_TOKENS,
                ),
            )
            if response.text:
                return response.text
        except Exception as err:
            last_err = err
            continue
    if last_err is not None:
        raise last_err
    return None

