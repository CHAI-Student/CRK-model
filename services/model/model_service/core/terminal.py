"""Terminal state restoration helpers for CLI entry points."""

from __future__ import annotations

import atexit
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, TextIO


@dataclass(frozen=True)
class TerminalState:
    """Original POSIX terminal attributes captured before the server starts."""

    fd: int
    attrs: list[Any]
    termios_module: Any


def capture_terminal_state(
    stream: TextIO | None = None,
    platform: str | None = None,
    termios_module: Any | None = None,
) -> TerminalState | None:
    """Capture current TTY attributes so they can be restored on shutdown."""

    current_platform = platform if platform is not None else os.name
    if current_platform != "posix":
        return None

    target_stream = stream if stream is not None else sys.stdin
    if target_stream is None or not hasattr(target_stream, "isatty"):
        return None
    if not target_stream.isatty():
        return None

    try:
        termios_ref = termios_module
        if termios_ref is None:
            import termios as termios_ref  # type: ignore[no-redef]

        fd = target_stream.fileno()
        attrs = termios_ref.tcgetattr(fd)
    except (AttributeError, ImportError, OSError, ValueError):
        return None
    except Exception as exc:
        if exc.__class__.__name__ == "error":
            return None
        raise

    return TerminalState(fd=fd, attrs=list(attrs), termios_module=termios_ref)


def restore_terminal_state(state: TerminalState | None) -> None:
    """Best-effort restoration of previously captured terminal attributes."""

    if state is None:
        return

    try:
        state.termios_module.tcsetattr(
            state.fd,
            state.termios_module.TCSADRAIN,
            state.attrs,
        )
    except (AttributeError, OSError, ValueError):
        return
    except Exception as exc:
        if exc.__class__.__name__ == "error":
            return
        raise


def install_terminal_restore(
    stream: TextIO | None = None,
    platform: str | None = None,
    termios_module: Any | None = None,
    atexit_register: Callable[..., Any] = atexit.register,
) -> TerminalState | None:
    """Capture terminal attributes and register an atexit restore callback."""

    state = capture_terminal_state(
        stream=stream,
        platform=platform,
        termios_module=termios_module,
    )
    if state is not None:
        atexit_register(restore_terminal_state, state)
    return state
