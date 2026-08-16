# Corrected Whisper for Hermes Agent

`corrected-whisper` is an STT provider for Hermes Agent gateway voice
messages. It combines two stages:

1. the native `local` provider, based on faster-whisper;
2. correction through an auxiliary `ctx.llm` call.

The plugin calls Hermes's local helper directly. It therefore retains the model
cache, locking, unloading after inactivity, CUDA-to-CPU fallback, VAD, and
anti-hallucination safeguards. It does not create an additional `WhisperModel`
instance.

## Installation

_Hermes >= 0.20.1_

### Install via Hermes Dashboard

Install in Plugins > Install from GitHub / Git URL > `agalbourdin/tools/hermes/plugins/corrected-whisper`

### Install via CLI

You'll need to restart the Gateway and the Dashboard after a CLI install.

```bash
hermes plugins install agalbourdin/tools/hermes/plugins/corrected-whisper --enable
```

## Configuration

Example:

```yaml
stt:
  enabled: true
  provider: corrected-whisper
  local:
    model: small
  # Other stt.local options remain supported

auxiliary:
  corrected_whisper_cleanup:
    provider: openai-codex
    model: gpt-5.6-luna
    reasoning_effort: low
```

## Notes

Version 0.2.0 adds a tab to the web dashboard. After the first installation or
an update, also restart `hermes dashboard` or rescan dashboard extensions.

If Whisper fails, the provider returns the native failure. If LLM correction
fails or returns empty text, the raw Whisper transcript is returned. An empty
or silent transcript does not trigger an LLM call.

The raw transcript is sent to the configured auxiliary LLM provider for
correction and may leave your machine.

The correction prompt can be changed from the **Corrected Whisper** dashboard
tab.

The prompt and transcript contents are never written to plugin logs.
