from __future__ import annotations

import pytest


class FakeStream:
    def __init__(self, tty: bool = True, fd: int = 7):
        self._tty = tty
        self._fd = fd

    def isatty(self) -> bool:
        return self._tty

    def fileno(self) -> int:
        return self._fd


class FakeTermios:
    TCSADRAIN = 1

    def __init__(self):
        self.restored: list[tuple[int, int, list[str]]] = []

    def tcgetattr(self, fd: int) -> list[str]:
        return ["original", str(fd)]

    def tcsetattr(self, fd: int, when: int, attrs: list[str]) -> None:
        self.restored.append((fd, when, attrs))


def test_capture_and_restore_terminal_state():
    from model_service.core.terminal import capture_terminal_state, restore_terminal_state

    fake_termios = FakeTermios()
    state = capture_terminal_state(
        stream=FakeStream(),
        platform="posix",
        termios_module=fake_termios,
    )

    assert state is not None
    restore_terminal_state(state)

    assert fake_termios.restored == [(7, fake_termios.TCSADRAIN, ["original", "7"])]


def test_capture_terminal_state_ignores_non_tty():
    from model_service.core.terminal import capture_terminal_state

    assert (
        capture_terminal_state(
            stream=FakeStream(tty=False),
            platform="posix",
            termios_module=FakeTermios(),
        )
        is None
    )


def test_install_terminal_restore_registers_atexit_callback():
    from model_service.core.terminal import install_terminal_restore

    callbacks = []
    fake_termios = FakeTermios()

    state = install_terminal_restore(
        stream=FakeStream(),
        platform="posix",
        termios_module=fake_termios,
        atexit_register=lambda callback, *args: callbacks.append((callback, args)),
    )

    assert state is not None
    assert len(callbacks) == 1

    callback, args = callbacks[0]
    callback(*args)

    assert fake_termios.restored == [(7, fake_termios.TCSADRAIN, ["original", "7"])]


def test_entrypoint_restores_terminal_when_interrupted(monkeypatch):
    import model_service.main as entrypoint

    restored = []
    monkeypatch.setattr(entrypoint, "install_terminal_restore", lambda: "saved-state")
    monkeypatch.setattr(entrypoint, "restore_terminal_state", lambda state: restored.append(state))
    monkeypatch.setattr(
        entrypoint,
        "main",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.run()

    assert exc_info.value.code == 0
    assert restored == ["saved-state"]
