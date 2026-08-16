from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class _HostTranscriptionProvider:
    """Minimal stand-in for Hermes's TranscriptionProvider ABC."""


class CorrectedWhisperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_modules = {
            name: sys.modules.get(name)
            for name in ("agent", "agent.transcription_provider", "tools", "tools.transcription_tools")
        }

        agent_package = types.ModuleType("agent")
        agent_package.__path__ = []
        transcription_provider = types.ModuleType("agent.transcription_provider")
        transcription_provider.TranscriptionProvider = _HostTranscriptionProvider
        agent_package.transcription_provider = transcription_provider

        tools_package = types.ModuleType("tools")
        tools_package.__path__ = []
        native_stt = types.ModuleType("tools.transcription_tools")
        native_stt.DEFAULT_LOCAL_MODEL = "base"
        tools_package.transcription_tools = native_stt

        sys.modules["agent"] = agent_package
        sys.modules["agent.transcription_provider"] = transcription_provider
        sys.modules["tools"] = tools_package
        sys.modules["tools.transcription_tools"] = native_stt
        cls.native_stt = native_stt

        plugin_path = Path(__file__).resolve().parents[1] / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "corrected_whisper_plugin_under_test", plugin_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load plugin from {plugin_path}")
        cls.plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.plugin)

    @classmethod
    def tearDownClass(cls) -> None:
        for name, previous in cls._saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def setUp(self) -> None:
        self.native_stt.DEFAULT_LOCAL_MODEL = "base"
        self.native_stt._load_stt_config = Mock(return_value={"local": {"model": "base"}})
        self.native_stt._normalize_local_model = Mock(side_effect=lambda value: value)
        self.native_stt._transcribe_local = Mock(
            return_value={
                "success": True,
                "transcript": "hello open ai",
                "provider": "local",
            }
        )
        self.llm = Mock()
        self.llm.complete.return_value = SimpleNamespace(text="Hello OpenAI.")
        self.get_config = Mock(return_value=self.plugin.DEFAULT_CLEANUP_PROMPT)
        self.provider = self.plugin.CorrectedWhisperProvider(
            self.llm,
            self.get_config,
        )

    def test_registers_auxiliary_task_before_provider(self) -> None:
        operations = []

        class FakeContext:
            llm = object()

            def get_config(self, key, default=None):
                return default

            def register_auxiliary_task(self, *args, **kwargs):
                operations.append(("auxiliary", args, kwargs))

            def register_transcription_provider(self, provider):
                operations.append(("provider", provider))

        context = FakeContext()
        self.plugin.register(context)

        self.assertEqual([item[0] for item in operations], ["auxiliary", "provider"])
        self.assertEqual(operations[0][1], ("corrected_whisper_cleanup",))
        self.assertEqual(operations[0][2]["display_name"], "STT cleanup")
        registered_provider = operations[1][1]
        self.assertIsInstance(registered_provider, _HostTranscriptionProvider)
        self.assertEqual(registered_provider.name, "corrected-whisper")
        self.assertIs(registered_provider._llm, context.llm)

    def test_success_uses_native_local_stt_then_task_routed_cleanup(self) -> None:
        result = self.provider.transcribe(
            "/tmp/voice.ogg",
            model="small",
            language="fr",
            prompt="OpenAI, Hermes",
        )

        self.native_stt._normalize_local_model.assert_called_once_with("small")
        self.native_stt._transcribe_local.assert_called_once_with(
            "/tmp/voice.ogg",
            "small",
            language="fr",
            prompt="OpenAI, Hermes",
        )
        self.assertEqual(result["transcript"], "Hello OpenAI.")
        self.assertEqual(result["provider"], "corrected-whisper")

        kwargs = self.llm.complete.call_args.kwargs
        self.assertEqual(kwargs["task"], "corrected_whisper_cleanup")
        self.assertEqual(kwargs["purpose"], "corrected-whisper.stt-cleanup")
        self.assertEqual(kwargs["messages"][1], {"role": "user", "content": "hello open ai"})
        self.assertIn("Do not summarize", kwargs["messages"][0]["content"])
        self.get_config.assert_called_once_with(
            "cleanup_prompt",
            self.plugin.DEFAULT_CLEANUP_PROMPT,
        )
        for forbidden_override in ("provider", "model", "profile", "agent_id"):
            self.assertNotIn(forbidden_override, kwargs)

    def test_local_language_overrides_forwarded_language(self) -> None:
        self.native_stt._load_stt_config.return_value = {
            "local": {"model": "base", "language": "fr"}
        }

        self.provider.transcribe("/tmp/voice.ogg", language="en")

        self.assertEqual(
            self.native_stt._transcribe_local.call_args.kwargs["language"],
            "fr",
        )

    def test_missing_language_preserves_automatic_detection(self) -> None:
        self.provider.transcribe("/tmp/voice.ogg")

        self.assertIsNone(
            self.native_stt._transcribe_local.call_args.kwargs["language"]
        )

    def test_configured_cleanup_prompt_is_used_verbatim(self) -> None:
        configured_prompt = "Correct transcription errors only.\nKeep every detail.\n"
        self.get_config.return_value = configured_prompt

        self.provider.transcribe("/tmp/voice.ogg")

        messages = self.llm.complete.call_args.kwargs["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": configured_prompt})

    def test_cleanup_prompt_is_reloaded_for_each_transcription(self) -> None:
        self.get_config.side_effect = ["First prompt", "Second prompt"]

        self.provider.transcribe("/tmp/first.ogg")
        self.provider.transcribe("/tmp/second.ogg")

        first_messages = self.llm.complete.call_args_list[0].kwargs["messages"]
        second_messages = self.llm.complete.call_args_list[1].kwargs["messages"]
        self.assertEqual(first_messages[0]["content"], "First prompt")
        self.assertEqual(second_messages[0]["content"], "Second prompt")

    def test_invalid_cleanup_prompt_falls_back_to_default(self) -> None:
        for invalid_prompt in (None, "", "  \n", {"unexpected": "mapping"}):
            with self.subTest(invalid_prompt=repr(invalid_prompt)):
                self.get_config.reset_mock()
                self.llm.complete.reset_mock()
                self.get_config.return_value = invalid_prompt

                self.provider.transcribe("/tmp/voice.ogg")

                messages = self.llm.complete.call_args.kwargs["messages"]
                self.assertEqual(
                    messages[0]["content"],
                    self.plugin.DEFAULT_CLEANUP_PROMPT,
                )

    def test_cleanup_prompt_read_failure_falls_back_to_default(self) -> None:
        self.get_config.side_effect = RuntimeError("config unavailable")

        self.provider.transcribe("/tmp/voice.ogg")

        messages = self.llm.complete.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], self.plugin.DEFAULT_CLEANUP_PROMPT)

    def test_model_falls_back_to_stt_local_configuration(self) -> None:
        self.native_stt._load_stt_config.return_value = {"local": {"model": "large-v3"}}

        self.provider.transcribe("/tmp/voice.ogg")

        self.native_stt._normalize_local_model.assert_called_once_with("large-v3")
        self.native_stt._transcribe_local.assert_called_once_with(
            "/tmp/voice.ogg",
            "large-v3",
            language=None,
            prompt=None,
        )

    def test_model_falls_back_to_native_default(self) -> None:
        self.native_stt._load_stt_config.return_value = {"local": None}

        self.provider.transcribe("/tmp/voice.ogg")

        self.native_stt._normalize_local_model.assert_called_once_with("base")

    def test_native_failure_is_preserved_without_cleanup(self) -> None:
        self.native_stt._transcribe_local.return_value = {
            "success": False,
            "transcript": "",
            "error": "faster-whisper not installed",
            "provider": "local",
        }

        result = self.provider.transcribe("/tmp/voice.ogg")

        self.assertEqual(result["error"], "faster-whisper not installed")
        self.assertEqual(result["transcript"], "")
        self.assertEqual(result["provider"], "corrected-whisper")
        self.get_config.assert_not_called()
        self.llm.complete.assert_not_called()

    def test_cleanup_exception_returns_raw_transcript(self) -> None:
        self.llm.complete.side_effect = RuntimeError("provider unavailable")

        result = self.provider.transcribe("/tmp/voice.ogg")

        self.assertTrue(result["success"])
        self.assertEqual(result["transcript"], "hello open ai")
        self.assertEqual(result["provider"], "corrected-whisper")

    def test_empty_cleanup_returns_raw_transcript(self) -> None:
        for empty_text in ("", "   \n"):
            with self.subTest(empty_text=repr(empty_text)):
                self.llm.complete.reset_mock()
                self.llm.complete.return_value = SimpleNamespace(text=empty_text)

                result = self.provider.transcribe("/tmp/voice.ogg")

                self.assertEqual(result["transcript"], "hello open ai")
                self.llm.complete.assert_called_once()

    def test_empty_native_transcript_skips_cleanup(self) -> None:
        for raw_text in ("", "  \n"):
            with self.subTest(raw_text=repr(raw_text)):
                self.llm.complete.reset_mock()
                self.native_stt._transcribe_local.return_value = {
                    "success": True,
                    "transcript": raw_text,
                    "provider": "local",
                }

                result = self.provider.transcribe("/tmp/voice.ogg")

                self.assertEqual(result["transcript"], raw_text)
                self.assertEqual(result["provider"], "corrected-whisper")
                self.get_config.assert_not_called()
                self.llm.complete.assert_not_called()

    def test_native_exception_becomes_standard_error_envelope(self) -> None:
        self.native_stt._transcribe_local.side_effect = RuntimeError("unexpected")

        result = self.provider.transcribe("/tmp/voice.ogg")

        self.assertFalse(result["success"])
        self.assertEqual(result["transcript"], "")
        self.assertEqual(result["provider"], "corrected-whisper")
        self.assertIn("unexpected", result["error"])
        self.get_config.assert_not_called()
        self.llm.complete.assert_not_called()

    def test_availability_tracks_native_helper_not_optional_dependency_flag(self) -> None:
        self.assertTrue(self.provider.is_available())
        del self.native_stt._transcribe_local
        self.assertFalse(self.provider.is_available())

    def test_dashboard_manifest_and_bundle_contract(self) -> None:
        plugin_dir = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (plugin_dir / "dashboard" / "manifest.json").read_text(encoding="utf-8")
        )
        bundle = (plugin_dir / "dashboard" / "dist" / "index.js").read_text(
            encoding="utf-8"
        )

        self.assertEqual(manifest["name"], "corrected-whisper")
        self.assertEqual(manifest["tab"]["path"], "/corrected-whisper")
        self.assertEqual(manifest["entry"], "dist/index.js")
        self.assertIn("SDK.api.getConfig()", bundle)
        self.assertIn("SDK.api.saveConfig", bundle)
        self.assertIn('const CONFIG_KEY = "cleanup_prompt"', bundle)
        self.assertIn(
            "window.__HERMES_PLUGINS__.register(PLUGIN_NAME, CorrectedWhisperPage)",
            bundle,
        )

    def test_default_prompt_asset_is_the_runtime_default(self) -> None:
        plugin_dir = Path(__file__).resolve().parents[1]
        defaults = json.loads(
            (plugin_dir / "dashboard" / "dist" / "defaults.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            defaults["cleanup_prompt"],
            self.plugin.DEFAULT_CLEANUP_PROMPT,
        )

    def test_plugin_manifest_is_installer_compatible_and_declares_prompt(self) -> None:
        plugin_dir = Path(__file__).resolve().parents[1]
        manifest_text = (plugin_dir / "plugin.yaml").read_text(encoding="utf-8")

        declarations = [
            line
            for line in manifest_text.splitlines()
            if line and not line.lstrip().startswith("#")
        ]
        self.assertFalse(
            any(line.startswith("manifest_version:") for line in declarations)
        )
        self.assertFalse(any(line.startswith("api_version:") for line in declarations))
        self.assertIn("config_schema:", manifest_text)
        self.assertIn("  cleanup_prompt:", manifest_text)
        self.assertIn("    type: str", manifest_text)


if __name__ == "__main__":
    unittest.main()
