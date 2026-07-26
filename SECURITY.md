# Security Policy

## Reporting a vulnerability

Report privately through GitHub's **Security → Report a vulnerability** (private vulnerability
reporting) on this repository. Please do not open a public issue for anything exploitable.

Include: affected version, platform, reproduction steps, and impact. Expect an acknowledgement
within 7 days and a fix or mitigation plan within 30 days for confirmed issues.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |
| < 0.1 | ❌ |

## How PlotaViz handles secrets

- **API keys are stored in the OS keyring** via the `keyring` package — macOS Keychain, or the
  Secret Service / kwallet backend on Linux. Keys are never written to config files, session
  files, logs, or the repository.
- Session files (`.pviz`) contain dataset paths, preprocessing steps, filters, and chart specs.
  **They never contain credentials.**
- Log output redacts anything that looks like a key before it is written.

## How PlotaViz handles your data

- The rules + scoring engine is **fully local**. No network access is required to load, clean,
  profile, or chart a dataset.
- The LLM layer is opt-in. When enabled, it sends **column schema, summary statistics, and a small
  sample of rows** — never the full dataset. The app asks for explicit consent before the first
  request in a session.
- The **Ollama provider requires no network egress at all** — it talks to a local model server.
  Use it for sensitive data.
- Model responses are treated as untrusted input: the LLM returns a JSON chart spec, which is
  validated against the actual dataframe schema before anything is rendered. The app never
  executes model-generated code.

## Scope

In scope: credential leakage, code execution via crafted datasets or session files, path traversal
on session load, and exfiltration of data beyond the documented LLM payload.

Out of scope: vulnerabilities in upstream dependencies with no PlotaViz-specific amplification
(report those upstream), and issues that require an already-compromised local machine.
