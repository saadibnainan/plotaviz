"""Provider abstraction and the prompt/validation machinery around it.

A provider's entire job is to turn two strings into one string. Everything else — building the
prompt, deciding what data is safe to send, extracting JSON from a chatty reply, validating the
returned spec against the real schema — happens here, once, so every provider behaves identically
and a new one is about thirty lines.

Two rules are structural, not stylistic:

**The model returns a chart spec, never code.** PlotaViz does not execute anything a model
produces. The reply is parsed as JSON and validated against the dataframe's actual columns; an
unknown column is a rejection with a clear message, not a traceback.

**The model sees schema, statistics, and a few sample rows.** Never the full dataset. This is a
privacy boundary and a token-cost boundary at the same time.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..errors import LLMError, ProviderNotConfigured
from ..profiler import DatasetProfile
from ..spec import CHART_TYPES, ChartSpec

#: Service name used for keyring entries. One entry per provider.
KEYRING_SERVICE = "plotaviz"

#: Default per-request timeout in seconds. LLM latency must never hang the UI.
DEFAULT_TIMEOUT = 30.0

#: Sample rows included in a prompt. Small on purpose.
SAMPLE_ROWS = 5

_SYSTEM_PROMPT = f"""You are the chart-selection assistant inside PlotaViz, a desktop data \
visualization tool.

You receive a dataset's schema and summary statistics — never the dataset itself. You reply with \
a single JSON object describing which chart to draw. You never write code, never write prose \
outside the JSON, and never invent column names.

Reply with exactly this shape:

{{
  "chart": one of {list(CHART_TYPES)},
  "x": column name or null,
  "y": column name or null,
  "color": column name or null,
  "agg": one of ["sum","mean","median","min","max","count","nunique"] or null,
  "title": short human title or null,
  "options": {{}},
  "why": one or two sentences explaining the choice in plain language
}}

Rules:
- Every column name you use MUST appear in the provided schema, spelled exactly.
- Use "agg" whenever y would otherwise have several values per x.
- Prefer readable charts: do not put a 500-category column on a bar axis.
- Columns marked "is_identifier": true are keys, not measures. Do not plot them.
- Respond with the JSON object alone. No markdown fences, no commentary.
"""


@dataclass
class LLMResult:
    """What the LLM layer returns to the app.

    Attributes:
        spec: The validated chart spec.
        narrative: Optional insight text the model offered alongside the choice.
        provider: Which provider answered.
        model: Which model answered.
        raw: The raw reply, kept for the "show me what it said" affordance.
    """

    spec: ChartSpec
    narrative: str = ""
    provider: str = ""
    model: str = ""
    raw: str = ""


class Provider(ABC):
    """Base class for an LLM provider.

    Subclasses implement :meth:`complete` and declare :attr:`name` and :attr:`default_model`.
    Everything else is inherited.

    Args:
        model: Model identifier. Falls back to :attr:`default_model`.
        api_key: Explicit key. When omitted, the key is read from the OS keyring.
        timeout: Per-request timeout in seconds.
    """

    #: Identifier used in settings and keyring entries.
    name: str = "base"

    #: Model used when the user has not chosen one.
    default_model: str = ""

    #: Models offered in the settings dialog.
    available_models: tuple[str, ...] = ()

    #: Whether this provider sends data off the machine. False for local runtimes.
    sends_data_off_machine: bool = True

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model or self.default_model
        self._api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------ credentials

    @property
    def api_key(self) -> str | None:
        """The API key, from the constructor or the OS keyring.

        Keys are never read from environment variables or config files by default — the keyring
        is the one storage location, so there is one place to audit.
        """
        if self._api_key:
            return self._api_key
        return get_api_key(self.name)

    def require_key(self) -> str:
        """Return the API key or explain how to set one.

        Raises:
            ProviderNotConfigured: If no key is stored.
        """
        key = self.api_key
        if not key:
            raise ProviderNotConfigured(
                f"No API key is stored for {self.name}.",
                hint="Add one in Settings → LLM provider. Keys are kept in your OS keychain.",
            )
        return key

    def is_configured(self) -> bool:
        """Whether this provider could make a request right now."""
        return bool(self.api_key)

    # ------------------------------------------------------------------ contract

    @abstractmethod
    def complete(self, system: str, user: str, *, timeout: float) -> str:
        """Send one prompt and return the raw text reply.

        Args:
            system: System prompt.
            user: User message.
            timeout: Seconds to wait before giving up.

        Raises:
            LLMError: On any transport, auth, or API failure.
        """

    # ------------------------------------------------------------------ tasks

    def suggest_chart(
        self,
        profile: DatasetProfile,
        *,
        question: str | None = None,
        candidates: list[ChartSpec] | None = None,
    ) -> LLMResult:
        """Ask the model for a chart spec.

        Serves both jobs the LLM layer exists for: with ``question`` it is natural-language →
        spec; without one it is a second opinion on an ambiguous ranking.

        Args:
            profile: The dataset profile. Only its schema summary is sent.
            question: The user's natural-language request, if any.
            candidates: The local engine's top candidates, sent as context so the model can agree
                with a good ranking rather than inventing something worse.

        Returns:
            A validated :class:`LLMResult`.

        Raises:
            LLMError: If the request fails or the reply is not a usable spec. Callers are
                expected to catch this and fall back to the local engine.
        """
        prompt = build_prompt(profile, question=question, candidates=candidates)
        raw = self.complete(_SYSTEM_PROMPT, prompt, timeout=self.timeout)
        payload = extract_json(raw)
        spec = ChartSpec.from_dict(payload)
        spec.source = "llm"
        spec.validate(list(profile.columns))
        return LLMResult(
            spec=spec,
            narrative=str(payload.get("narrative") or payload.get("insight") or ""),
            provider=self.name,
            model=self.model,
            raw=raw,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


# ---------------------------------------------------------------------------- prompts


def build_prompt(
    profile: DatasetProfile,
    *,
    question: str | None = None,
    candidates: list[ChartSpec] | None = None,
    sample_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Assemble the user message.

    Contains the schema, per-column statistics, at most :data:`SAMPLE_ROWS` example rows, and the
    local engine's candidates. Deliberately does not contain the dataset.
    """
    summary = profile.schema_summary()
    parts = [
        "Dataset schema and summary statistics:",
        json.dumps(summary, indent=2, default=str),
    ]

    if sample_rows:
        parts += [
            "",
            f"A sample of {min(len(sample_rows), SAMPLE_ROWS)} rows (illustrative only):",
            json.dumps(sample_rows[:SAMPLE_ROWS], indent=2, default=str),
        ]

    if candidates:
        ranked = [
            {"chart": c.chart, "x": c.x, "y": c.y, "color": c.color, "score": round(c.score, 3)}
            for c in candidates[:5]
        ]
        parts += [
            "",
            "PlotaViz's own rules engine ranked these candidates. Their scores were close, "
            "which is why you are being asked:",
            json.dumps(ranked, indent=2),
        ]

    if question:
        parts += [
            "",
            f"The user asked for: {question!r}",
            "",
            "Return the chart spec that answers it.",
        ]
    else:
        parts += ["", "Return the single chart spec that best reveals what is in this dataset."]

    return "\n".join(parts)


def sample_rows_for(df: Any, n: int = SAMPLE_ROWS) -> list[dict[str, Any]]:
    """Take a few example rows in a JSON-safe form.

    Kept tiny by design — enough for the model to see value formats, not enough to constitute
    sending someone's data anywhere.
    """
    try:
        head = df.head(n)
        return json.loads(head.to_json(orient="records", date_format="iso")) or []
    except Exception:
        return []


def extract_json(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a model reply.

    Models add markdown fences and preambles no matter how firmly they are told not to, so this
    tolerates both rather than failing on formatting.

    Raises:
        LLMError: If no JSON object can be found or parsed.
    """
    if not text or not text.strip():
        raise LLMError("The model returned an empty response.")

    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(
                "The model's reply did not contain a chart specification.",
                hint="Try rephrasing, or use the ranked recommendations from the local engine.",
            ) from None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError("The model's reply was not valid JSON.", hint=str(exc)) from exc

    if not isinstance(payload, dict):
        raise LLMError("The model returned JSON, but not a chart specification object.")
    return payload


# ---------------------------------------------------------------------------- keyring


def get_api_key(provider: str) -> str | None:
    """Read a provider's API key from the OS keyring.

    Returns ``None`` when nothing is stored or no keyring backend is available — a missing key is
    an ordinary state, since PlotaViz works fully offline.
    """
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, provider)
    except Exception:
        return None


def set_api_key(provider: str, key: str) -> None:
    """Store a provider's API key in the OS keyring.

    Raises:
        LLMError: If no keyring backend is available. PlotaViz does not fall back to writing the
            key to a file — that is exactly the failure mode the keyring exists to prevent.
    """
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, provider, key)
    except Exception as exc:
        raise LLMError(
            "Could not store the API key in your system keychain.",
            hint=(
                "PlotaViz will not write keys to disk in plain text. On Linux, install a Secret "
                f"Service backend such as gnome-keyring or kwallet. Underlying error: {exc}"
            ),
        ) from exc


def delete_api_key(provider: str) -> None:
    """Remove a provider's stored API key. Missing entries are not an error."""
    try:
        import keyring

        keyring.delete_password(KEYRING_SERVICE, provider)
    except Exception:
        pass


def redact(text: str) -> str:
    """Mask anything that looks like an API key before it reaches a log or an error dialog."""
    patterns = (
        r"sk-ant-[A-Za-z0-9\-_]{8,}",
        r"sk-[A-Za-z0-9]{16,}",
        r"AIza[0-9A-Za-z\-_]{20,}",
        r"Bearer\s+[A-Za-z0-9\-._~+/]{16,}",
    )
    out = text
    for pattern in patterns:
        out = re.sub(pattern, "[redacted]", out)
    return out
