"""Session files — ``.pviz`` projects.

A session stores the *recipe*, never the data: source path and hash, type overrides, the
preprocessing step list, active filters, the chart spec, and view settings. Reopening replays the
recipe against the source file.

That design has one consequence worth being explicit about: if the source file changed after the
session was saved, the replay may produce something different. The stored SHA-256 is checked on
load and the mismatch is reported rather than swallowed.

Sessions never contain credentials. API keys live in the OS keyring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import SessionError
from .loader import file_hash
from .preprocess import Pipeline
from .spec import ChartSpec

#: Bump when the on-disk shape changes incompatibly.
SESSION_VERSION = 1

#: Session file extension.
SESSION_SUFFIX = ".pviz"


@dataclass
class Session:
    """A saved PlotaViz project.

    Attributes:
        source_path: Dataset the session was built from.
        source_hash: SHA-256 of that file at save time.
        type_overrides: ``{column: role}`` corrections the user made.
        pipeline: The preprocessing steps, including filters.
        spec: The chart that was on screen.
        view: View settings — window geometry, active panel, renderer choice.
        app_version: PlotaViz version that wrote the file.
        saved_at: ISO-8601 UTC timestamp.
        version: Session schema version.
    """

    source_path: str
    source_hash: str = ""
    type_overrides: dict[str, str] = field(default_factory=dict)
    pipeline: Pipeline = field(default_factory=Pipeline)
    spec: ChartSpec | None = None
    view: dict[str, Any] = field(default_factory=dict)
    app_version: str = "0.1.0"
    saved_at: str = ""
    version: int = SESSION_VERSION

    # ------------------------------------------------------------------ writing

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON structure written to disk."""
        return {
            "version": self.version,
            "app_version": self.app_version,
            "saved_at": self.saved_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "source": {"path": str(self.source_path), "sha256": self.source_hash},
            "type_overrides": dict(self.type_overrides),
            "pipeline": self.pipeline.to_list(),
            "chart": self.spec.to_dict() if self.spec else None,
            "view": dict(self.view),
        }

    def save(self, path: str | Path) -> Path:
        """Write the session to ``path``, adding the ``.pviz`` suffix if absent.

        Raises:
            SessionError: If the file cannot be written.
        """
        target = Path(path).expanduser()
        if target.suffix != SESSION_SUFFIX:
            target = target.with_suffix(SESSION_SUFFIX)

        source = Path(self.source_path)
        if source.exists() and not self.source_hash:
            self.source_hash = file_hash(source)
        self.saved_at = datetime.now(UTC).isoformat(timespec="seconds")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except OSError as exc:
            raise SessionError(f"Could not save the session to {target}.", hint=str(exc)) from exc
        return target

    # ------------------------------------------------------------------ reading

    @classmethod
    def load(cls, path: str | Path) -> Session:
        """Read a ``.pviz`` file.

        Raises:
            SessionError: If the file is missing, is not valid JSON, or was written by a newer
                incompatible version.
        """
        target = Path(path).expanduser()
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SessionError(f"No session file at {target}.") from exc
        except json.JSONDecodeError as exc:
            raise SessionError(
                f"{target.name} is not a valid PlotaViz session file.", hint=str(exc)
            ) from exc
        except OSError as exc:
            raise SessionError(f"Could not read {target}.", hint=str(exc)) from exc

        version = int(payload.get("version", 0))
        if version > SESSION_VERSION:
            raise SessionError(
                f"{target.name} was saved by a newer version of PlotaViz "
                f"(session format {version}, this build understands {SESSION_VERSION}).",
                hint="Update PlotaViz to open it.",
            )

        source = payload.get("source") or {}
        chart = payload.get("chart")

        try:
            pipeline = Pipeline.from_list(payload.get("pipeline") or [])
        except Exception as exc:
            raise SessionError(
                f"The preprocessing steps in {target.name} could not be restored.",
                hint=str(exc),
            ) from exc

        return cls(
            source_path=str(source.get("path", "")),
            source_hash=str(source.get("sha256", "")),
            type_overrides=dict(payload.get("type_overrides") or {}),
            pipeline=pipeline,
            spec=ChartSpec.from_dict(chart) if chart else None,
            view=dict(payload.get("view") or {}),
            app_version=str(payload.get("app_version", "")),
            saved_at=str(payload.get("saved_at", "")),
            version=version,
        )

    # ------------------------------------------------------------------ validation

    def check_source(self) -> str | None:
        """Verify the source file is where it was and unchanged.

        Returns:
            ``None`` when everything matches, otherwise a user-facing warning describing the
            discrepancy. The caller decides whether to continue — usually it should, because a
            changed source is normal and only worth mentioning.
        """
        source = Path(self.source_path)
        if not self.source_path:
            return "This session does not record which file it came from."
        if not source.exists():
            return (
                f"The source file {source} is missing. Open the dataset again to restore this "
                "session."
            )
        if self.source_hash and file_hash(source) != self.source_hash:
            saved = self.saved_at or "the time it was saved"
            return (
                f"{source.name} has changed since {saved}. The cleaning steps and chart will be "
                "replayed against the new contents, so results may differ."
            )
        return None
