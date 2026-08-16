(function () {
  "use strict";

  const PLUGIN_NAME = "corrected-whisper";
  const CONFIG_KEY = "cleanup_prompt";
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { useCallback, useEffect, useState } = SDK.hooks;
  const {
    Button,
    Card,
    CardContent,
    CardHeader,
    CardTitle,
  } = SDK.components;

  function configuredPrompt(config) {
    const value = config?.plugins?.entries?.[PLUGIN_NAME]?.settings?.[CONFIG_KEY];
    return typeof value === "string" && value.trim() ? value : null;
  }

  function configUpdate(prompt) {
    return {
      plugins: {
        entries: {
          [PLUGIN_NAME]: {
            settings: {
              [CONFIG_KEY]: prompt,
            },
          },
        },
      },
    };
  }

  function CorrectedWhisperPage() {
    const [prompt, setPrompt] = useState("");
    const [savedPrompt, setSavedPrompt] = useState("");
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState("");
    const [error, setError] = useState("");

    useEffect(() => {
      let active = true;
      Promise.all([
        SDK.fetchJSON(
          "/dashboard-plugins/corrected-whisper/dist/defaults.json",
        ),
        SDK.api.getConfig(),
      ])
        .then(([defaults, config]) => {
          if (!active) return;
          const fallback = defaults?.[CONFIG_KEY];
          if (typeof fallback !== "string" || !fallback.trim()) {
            throw new Error("The plugin's default prompt is invalid.");
          }
          const effective = configuredPrompt(config) || fallback;
          setPrompt(effective);
          setSavedPrompt(effective);
        })
        .catch((loadError) => {
          if (!active) return;
          setError(loadError?.message || "Unable to load the configuration.");
        })
        .finally(() => {
          if (active) setLoading(false);
        });
      return () => {
        active = false;
      };
    }, []);

    const persist = useCallback(async (nextPrompt, successMessage) => {
      if (typeof nextPrompt !== "string" || !nextPrompt.trim()) {
        setMessage("");
        setError("The prompt cannot be empty.");
        return;
      }
      setSaving(true);
      setMessage("");
      setError("");
      try {
        await SDK.api.saveConfig(configUpdate(nextPrompt));
        setPrompt(nextPrompt);
        setSavedPrompt(nextPrompt);
        setMessage(successMessage);
      } catch (saveError) {
        setError(saveError?.message || "Unable to save the prompt.");
      } finally {
        setSaving(false);
      }
    }, []);

    const isDirty = prompt !== savedPrompt;

    return React.createElement(
      "div",
      { style: { display: "grid", gap: "1rem", maxWidth: "960px" } },
      React.createElement(
        "div",
        null,
        React.createElement(
          "h1",
          { className: "font-display text-2xl tracking-wide" },
          "Corrected Whisper",
        ),
        React.createElement(
          "p",
          { className: "mt-1 text-sm text-text-secondary" },
          "Configure the prompt sent to the corrected_whisper_cleanup auxiliary task.",
        ),
      ),
      React.createElement(
        Card,
        null,
        React.createElement(
          CardHeader,
          null,
          React.createElement(CardTitle, null, "Correction prompt"),
        ),
        React.createElement(
          CardContent,
          null,
          React.createElement(
            "div",
            {
              style: {
                border: "1px solid var(--color-accent-foreground)",
                marginBottom: "1rem",
                padding: "0.75rem",
              },
            },
            React.createElement(
              "p",
              { className: "text-sm text-text-secondary" },
              "This field replaces the complete system prompt. Changing it may " +
                "weaken safeguards against summaries, additions, or deletions.",
            ),
          ),
          loading
            ? React.createElement(
                "p",
                { className: "text-sm text-text-secondary" },
                "Loading…",
              )
            : React.createElement("textarea", {
                "aria-label": "STT correction prompt",
                disabled: saving,
                onChange: (event) => {
                  setPrompt(event.target.value);
                  setMessage("");
                  setError("");
                },
                spellCheck: false,
                style: {
                  background: "var(--color-background)",
                  border: "1px solid var(--color-border)",
                  color: "var(--color-foreground)",
                  fontFamily: "var(--font-mono, ui-monospace, monospace)",
                  fontSize: "0.875rem",
                  lineHeight: "1.5",
                  minHeight: "24rem",
                  padding: "0.75rem",
                  resize: "vertical",
                  width: "100%",
                },
                value: prompt,
              }),
          error && React.createElement(
            "p",
            { role: "alert", className: "mt-3 text-sm text-destructive" },
            error,
          ),
          message && React.createElement(
            "p",
            { role: "status", className: "mt-3 text-sm text-success" },
            message,
          ),
          React.createElement(
            "div",
            {
              style: {
                display: "flex",
                flexWrap: "wrap",
                gap: "0.75rem",
                marginTop: "1rem",
              },
            },
            React.createElement(
              Button,
              {
                disabled: loading || saving || !isDirty || !prompt.trim(),
                onClick: () => persist(prompt, "Prompt saved."),
              },
              saving ? "Saving…" : "Save",
            ),
          ),
        ),
      ),
    );
  }

  window.__HERMES_PLUGINS__.register(PLUGIN_NAME, CorrectedWhisperPage);
})();
