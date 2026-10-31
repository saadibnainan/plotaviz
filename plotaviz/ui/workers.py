"""Background work — keeping long operations off the UI thread.

Loading a 400 MB CSV, replaying a pipeline, and waiting on an LLM all take long enough to freeze
a window, and a frozen window looks like a crash. Every one of those runs through :class:`Worker`
on a ``QThread``, reports progress, and can be cancelled.

The cancellation model is cooperative and honest: :meth:`Worker.cancel` sets a flag the task can
check, and if the task is stuck inside pandas the flag is only honoured when it returns. The UI
therefore says "Cancelling…" rather than pretending the work stopped instantly.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..core.errors import PlotaVizError


class Worker(QObject):
    """Runs one callable on a background thread.

    The callable may accept a ``progress`` keyword — a ``Callable[[int, str], None]`` it can call
    to report percentage and a status message — and a ``should_cancel`` keyword returning whether
    the user has asked to stop.

    Signals:
        finished: Emitted with the return value on success.
        failed: Emitted with a user-facing message and the traceback on failure.
        aborted: Emitted instead of ``finished`` when the user cancelled; the result is dropped.
        progress: Emitted with ``(percent, message)``.
        done: Emitted after any outcome, for teardown.
    """

    finished = Signal(object)
    failed = Signal(str, str)
    aborted = Signal()
    progress = Signal(int, str)
    done = Signal()

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the task to stop at its next checkpoint."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled

    def run(self) -> None:
        """Execute the callable. Connected to the thread's ``started`` signal."""
        try:
            import inspect

            params = inspect.signature(self._fn).parameters
            kwargs = dict(self._kwargs)
            if "progress" in params:
                kwargs["progress"] = self._emit_progress
            if "should_cancel" in params:
                kwargs["should_cancel"] = lambda: self._cancelled

            result = self._fn(*self._args, **kwargs)
        except PlotaVizError as exc:
            # Already written for a human — pass the message straight through.
            self.failed.emit(str(exc), traceback.format_exc())
        except Exception as exc:
            self.failed.emit(
                f"Something went wrong: {exc}",
                traceback.format_exc(),
            )
        else:
            # Every run ends in exactly one of finished / failed / cancelled. Emitting nothing on
            # cancel left the caller's progress bar, Cancel button, and busy state up forever.
            if self._cancelled:
                self.aborted.emit()
            else:
                self.finished.emit(result)
        finally:
            self.done.emit()

    def _emit_progress(self, percent: int, message: str = "") -> None:
        """Progress callback handed to the task."""
        self.progress.emit(int(percent), message)


class TaskRunner(QObject):
    """Owns a worker and its thread, and marshals results back to the UI thread.

    Two problems this solves, both of which crash an app rather than merely annoying it:

    * Qt garbage-collects a running ``QThread`` if nothing holds a reference, which shows up as
      an intermittent segfault under load. This object holds that reference.
    * A signal connected to a *plain callable* — a lambda, a nested function — is delivered
      synchronously **in the emitting thread**. Callbacks written against the UI would then touch
      widgets from the worker thread, which Qt does not allow. So worker signals are connected to
      slots on this object (which lives in the UI thread, so Qt queues the delivery), and those
      slots re-emit on the UI thread where the caller's callbacks are safe to run.

    Args:
        parent: Parent object, usually the main window.

    Signals:
        finished: Emitted on the UI thread with the task's return value.
        failed: Emitted on the UI thread with ``(message, traceback)``.
        cancelled: Emitted on the UI thread when a cancelled task has wound down.
        progress: Emitted on the UI thread with ``(percent, message)``.
        idle: Emitted on the UI thread once the thread has stopped; start queued work here.
    """

    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()
    progress = Signal(int, str)
    #: Emitted once the thread has fully stopped and a new task can start.
    idle = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._on_finished: Callable[[Any], None] | None = None
        self._on_failed: Callable[[str, str], None] | None = None
        self._on_cancelled: Callable[[], None] | None = None
        self._on_progress: Callable[[int, str], None] | None = None

    @property
    def busy(self) -> bool:
        """Whether a task is running or its results have not been delivered yet.

        Busy until :meth:`_clear` runs, not merely until the thread stops: the result is
        relayed to the UI thread through a queued signal, and a new task started in between used
        to replace the callbacks, so the *old* result reached the *new* caller.
        """
        return self._thread is not None

    def start(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_finished: Callable[[Any], None] | None = None,
        on_failed: Callable[[str, str], None] | None = None,
        on_progress: Callable[[int, str], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> Worker:
        """Run ``fn`` on a background thread.

        Args:
            fn: The callable to run.
            *args: Positional arguments for it.
            on_finished: Called with the result on the UI thread.
            on_failed: Called with ``(message, traceback)`` on the UI thread.
            on_progress: Called with ``(percent, message)`` on the UI thread.
            on_cancelled: Called on the UI thread when a cancelled task has finished running.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            The :class:`Worker`, so the caller can cancel it.

        Raises:
            RuntimeError: If a task is already running. Callers disable their triggers instead of
                relying on this.
        """
        if self.busy:
            raise RuntimeError("A background task is already running.")

        thread = QThread()
        worker = Worker(fn, *args, **kwargs)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)

        # Route through this object's slots so delivery is queued onto the UI thread; only then
        # are the caller's plain callables safe to invoke.
        self._on_finished = on_finished
        self._on_failed = on_failed
        self._on_progress = on_progress
        self._on_cancelled = on_cancelled
        worker.finished.connect(self._relay_finished)
        worker.failed.connect(self._relay_failed)
        worker.aborted.connect(self._relay_cancelled)
        worker.progress.connect(self._relay_progress)

        worker.done.connect(thread.quit)
        # The worker is NOT deleteLater()'d on its own thread. Its Python wrapper would then be
        # destroyed there, taking the GIL while the UI thread may be inside a Qt call that has
        # released the GIL and needs it back for a Python slot — an intermittent deadlock seen
        # in testing. Instead the UI thread drops its reference in _clear(), after the thread has
        # stopped, which is when Qt allows deleting an object of a finished thread.
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear)

        self._thread, self._worker = thread, worker
        thread.start()
        return worker

    # ------------------------------------------------------------------ UI-thread relays

    @Slot(object)
    def _relay_finished(self, result: Any) -> None:
        """Runs on the UI thread. Re-emit, then call the caller's success handler."""
        self.finished.emit(result)
        if self._on_finished is not None:
            self._on_finished(result)

    @Slot(str, str)
    def _relay_failed(self, message: str, trace: str) -> None:
        """Runs on the UI thread. Re-emit, then call the caller's failure handler."""
        self.failed.emit(message, trace)
        if self._on_failed is not None:
            self._on_failed(message, trace)

    @Slot()
    def _relay_cancelled(self) -> None:
        """Runs on the UI thread. Re-emit, then call the caller's cancellation handler."""
        self.cancelled.emit()
        if self._on_cancelled is not None:
            self._on_cancelled()

    @Slot(int, str)
    def _relay_progress(self, percent: int, message: str) -> None:
        """Runs on the UI thread. Re-emit, then call the caller's progress handler."""
        self.progress.emit(percent, message)
        if self._on_progress is not None:
            self._on_progress(percent, message)

    def cancel(self) -> None:
        """Ask the running task to stop."""
        if self._worker is not None:
            self._worker.cancel()

    def wait(self, timeout_ms: int = 5000) -> None:
        """Block until the current task finishes. Used on window close."""
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(timeout_ms)

    @Slot()
    def _clear(self) -> None:
        """Drop references and per-task callbacks once the thread is done.

        Ignores a stale notification: if a new task was started after the old thread stopped
        running but before this queued slot fired, clearing would orphan the new task. (The
        finished thread may already be deleted by then, so it is not inspected — only whether
        the thread currently held is still running.)
        """
        try:
            if self._thread is not None and self._thread.isRunning():
                return
        except RuntimeError:  # the C++ object behind the held thread is already gone
            pass
        self._thread = None
        self._worker = None
        self._on_finished = None
        self._on_failed = None
        self._on_cancelled = None
        self._on_progress = None
        self.idle.emit()
