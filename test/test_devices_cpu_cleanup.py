import types

from modules import devices


def test_cpu_gc_runs_python_gc_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(devices.shared, "opts", types.SimpleNamespace(cpu_gc_after_generation=True, cpu_gc_malloc_trim=True), raising=False)
    monkeypatch.setattr(devices.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(devices, "malloc_trim", lambda: calls.append("trim") or True)

    devices.cpu_gc()

    assert calls == ["collect", "trim"]


def test_cpu_gc_skips_when_disabled_without_force(monkeypatch):
    calls = []
    monkeypatch.setattr(devices.shared, "opts", types.SimpleNamespace(cpu_gc_after_generation=False, cpu_gc_malloc_trim=True), raising=False)
    monkeypatch.setattr(devices.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(devices, "malloc_trim", lambda: calls.append("trim") or True)

    devices.cpu_gc()

    assert calls == []


def test_cpu_gc_force_runs_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(devices.shared, "opts", types.SimpleNamespace(cpu_gc_after_generation=False, cpu_gc_malloc_trim=False), raising=False)
    monkeypatch.setattr(devices.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(devices, "malloc_trim", lambda: calls.append("trim") or True)

    devices.cpu_gc(force=True)

    assert calls == ["collect"]


def test_malloc_trim_returns_false_on_non_linux(monkeypatch):
    monkeypatch.setattr(devices.sys, "platform", "win32")

    assert devices.malloc_trim() is False


def test_malloc_trim_returns_false_when_libc_unavailable(monkeypatch):
    monkeypatch.setattr(devices.sys, "platform", "linux")
    monkeypatch.setattr(devices.ctypes, "CDLL", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing libc")))

    assert devices.malloc_trim() is False
