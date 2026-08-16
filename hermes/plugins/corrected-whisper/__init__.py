"""Hermes STT provider: native local Whisper followed by conservative cleanup."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agent.transcription_provider import TranscriptionProvider
from tools import transcription_tools as native_stt

logger = logging.getLogger(__name__)

PROVIDER_NAME = "corrected-whisper"
CLEANUP_TASK = "corrected_whisper_cleanup"
CLEANUP_PROMPT_CONFIG_KEY = "cleanup_prompt"


def _load_default_cleanup_prompt() -> str:
    defaults_path = Path(__file__).parent / "dashboard" / "dist" / "defaults.json"
    with defaults_path.open(encoding="utf-8") as defaults_file:
        defaults = json.load(defaults_file)
    prompt = defaults.get(CLEANUP_PROMPT_CONFIG_KEY)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"{defaults_path} must contain a non-empty "
            f"{CLEANUP_PROMPT_CONFIG_KEY!r} string"
        )
    return prompt


DEFAULT_CLEANUP_PROMPT = _load_default_cleanup_prompt()


class CorrectedWhisperProvider(TranscriptionProvider):
    """Compose Hermes's native local STT with a host-routed cleanup call."""

    def __init__(self, llm: Any, get_config: Any = None) -> None:
        self._llm = llm
        self._get_config = get_config or (lambda _key, default=None: default)

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "Corrected Whisper"

    def is_available(self) -> bool:
        # The native helper owns faster-whisper availability and lazy install.
        return callable(getattr(native_stt, "_transcribe_local", None))

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe locally, then best-effort correct the resulting text."""
        try:
            stt_config = native_stt._load_stt_config()
            local_config = stt_config.get("local") if isinstance(stt_config, dict) else None
            if not isinstance(local_config, dict):
                local_config = {}
            model_name = native_stt._normalize_local_model(
                model or local_config.get("model", native_stt.DEFAULT_LOCAL_MODEL)
            )
            local_language = local_config.get("language")
            if not isinstance(local_language, str) or not local_language.strip():
                local_language = None
            raw_result = native_stt._transcribe_local(
                file_path,
                model_name,
                language=local_language or language,
                prompt=extra.get("prompt"),
            )
        except Exception as exc:  # noqa: BLE001 - provider contract is an envelope
            logger.error(
                "corrected-whisper local transcription failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return {
                "success": False,
                "transcript": "",
                "error": f"Corrected Whisper local transcription failed: {exc}",
                "provider": PROVIDER_NAME,
            }

        result = dict(raw_result)
        result["provider"] = PROVIDER_NAME
        if not result.get("success"):
            return result

        raw_transcript = result.get("transcript", "")
        if not isinstance(raw_transcript, str):
            raw_transcript = str(raw_transcript or "")
            result["transcript"] = raw_transcript
        if not raw_transcript.strip():
            return result

        cleanup_prompt = self._resolve_cleanup_prompt()
        try:
            cleanup = self._llm.complete(
                messages=[
                    {"role": "system", "content": cleanup_prompt},
                    {"role": "user", "content": raw_transcript},
                ],
                task=CLEANUP_TASK,
                purpose="corrected-whisper.stt-cleanup",
            )
            corrected = getattr(cleanup, "text", "")
            if not isinstance(corrected, str) or not corrected.strip():
                logger.warning(
                    "corrected-whisper cleanup returned empty text; using raw transcript"
                )
                return result
            result["transcript"] = corrected.strip()
            return result
        except Exception as exc:  # noqa: BLE001 - cleanup must fail open
            logger.warning(
                "corrected-whisper cleanup failed; using raw transcript (%s)",
                type(exc).__name__,
            )
            return result

    def _resolve_cleanup_prompt(self) -> str:
        """Read the effective prompt without exposing its contents in logs."""
        try:
            configured = self._get_config(
                CLEANUP_PROMPT_CONFIG_KEY,
                DEFAULT_CLEANUP_PROMPT,
            )
        except Exception as exc:  # noqa: BLE001 - configuration is best-effort
            logger.warning(
                "corrected-whisper could not read cleanup prompt; using default "
                "(%s)",
                type(exc).__name__,
            )
            return DEFAULT_CLEANUP_PROMPT

        if not isinstance(configured, str) or not configured.strip():
            logger.warning(
                "corrected-whisper cleanup prompt is empty or invalid; using default"
            )
            return DEFAULT_CLEANUP_PROMPT
        return configured


def register(ctx: Any) -> None:
    """Register the auxiliary route before exposing the STT provider."""
    ctx.register_auxiliary_task(
        CLEANUP_TASK,
        display_name="STT cleanup",
        description="Conservatively correct a local Whisper transcription.",
    )
    ctx.register_transcription_provider(
        CorrectedWhisperProvider(ctx.llm, ctx.get_config)
    )
