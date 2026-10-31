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
        load_options: Import settings the user chose (delimiter, encoding, decimal mark, sheet),
            replayed on restore. Empty means "detect", as for a normal open.
        type_overrides: ``{column: role}`` corrections the user made.
        pipeline: The preprocessing steps, including filters.
        spec: The chart that was on screen.
        view: View settings — window geometry, active panel, renderer choice.
        dashboard: Charts pinned to the dashboard, in tile order. Optional on disk, so files
            written before dashboards existed still load.
        app_version: PlotaViz version that wrote the file.
        saved_at: ISO-8601 UTC timestamp.
        version: Session schema version.
    """

    source_path: str
    source_hash: str = ""
    load_options: dict[str, Any] = field(default_factory=dict)
    type_overrides: dict[str, str] = field(default_factory=dict)
    pipeline: Pipeline = field(default_factory=Pipeline)
    spec: ChartSpec | None = None
    view: dict[str, Any] = field(default_factory=dict)
    dashboard: list[ChartSpec] = field(default_factory=list)
    app_version: str = "0.1.0"
    saved_at: str = ""
    version: int = SESSION_VERSION
    #: Hash of the source as found by the last :meth:`check_source`, so the loader can reuse it
    #: instead of reading a multi-gigabyte file a second time. Never written to disk.
    current_hash: str = field(default="", repr=False, compare=False)
    #: The source path relative to the ``.pviz`` file, so a project folder can be moved or
    #: shared and still find its data.
    source_relative: str = ""
    #: The recorded path the data was *not* found at, when it was found somewhere else.
    relocated_from: str = field(default="", repr=False, compare=False)

    # ------------------------------------------------------------------ writing

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON structure written to disk."""
        return {
            "version": self.version,
            "app_version": self.app_version,
            "saved_at": self.saved_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "source": {
                "path": str(self.source_path),
                "relative": self.source_relative,
                "sha256": self.source_hash,
                "options": dict(self.load_options),
            },
            "type_overrides": dict(self.type_overrides),
            "pipeline": self.pipeline.to_list(),
            "chart": self.spec.to_dict() if self.spec else None,
            "view": dict(self.view),
            "dashboard": [tile.to_dict() for tile in self.dashboard],
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
        self.source_relative = _relative_to(source, target.parent)
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

        session = cls(
            source_path=str(source.get("path", "")),
            source_relative=str(source.get("relative", "") or ""),
            source_hash=str(source.get("sha256", "")),
            load_options=dict(source.get("options") or {}),
            type_overrides=dict(payload.get("type_overrides") or {}),
            pipeline=pipeline,
            spec=ChartSpec.from_dict(chart) if chart else None,
            view=dict(payload.get("view") or {}),
            dashboard=[
                ChartSpec.from_dict(tile)
                for tile in payload.get("dashboard") or []
                if isinstance(tile, dict)
            ],
            app_version=str(payload.get("app_version", "")),
            saved_at=str(payload.get("saved_at", "")),
            version=version,
        )
        # A moved or shared project: the recorded absolute path is gone, but the data sits at the
        # same place relative to the session file. Use it, and say so in check_source().
        if session.source_relative and not Path(session.source_path).exists():
            candidate = (target.parent / session.source_relative).resolve()
            if candidate.exists():
                session.relocate(candidate)
        return session

    def relocate(self, path: str | Path) -> None:
        """Point the session at its data file's new location.

        The stored hash is kept, so :meth:`check_source` still reports whether the file at the
        new location has the same contents.
        """
        if not self.relocated_from:
            self.relocated_from = str(self.source_path)
        self.source_path = str(Path(path).expanduser().resolve())

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
        if self.source_hash:
            self.current_hash = file_hash(source)
        notes: list[str] = []
        if self.relocated_from:
            notes.append(f"The data file was not at {self.relocated_from}; using {source} instead.")
        if self.source_hash and self.current_hash != self.source_hash:
            saved = self.saved_at or "the time it was saved"
            notes.append(
                f"{source.name} has changed since {saved}. The cleaning steps and chart will be "
                "replayed against the new contents, so results may differ."
            )
        return "\n\n".join(notes) or None


def _relative_to(source: Path, folder: Path) -> str:
    """``source`` relative to ``folder``, or ``""`` when there is no relative path (another
    drive on Windows) or either path cannot be resolved."""
    import os

    try:
        return os.path.relpath(source.expanduser().resolve(), folder.expanduser().resolve())
    except (ValueError, OSError):
        return ""
