from types import SimpleNamespace

from modules import call_queue


def disable_memmon(monkeypatch):
    monkeypatch.setattr(call_queue.shared, "opts", SimpleNamespace(memmon_poll_rate=0), raising=False)
    monkeypatch.setattr(call_queue.shared, "mem_mon", SimpleNamespace(disabled=True), raising=False)


def test_gradio_call_runs_cpu_gc_after_success(monkeypatch):
    calls = []
    disable_memmon(monkeypatch)
    monkeypatch.setattr(call_queue.devices, "torch_gc", lambda: calls.append("torch"))
    monkeypatch.setattr(call_queue.devices, "cpu_gc", lambda force=False: calls.append(("cpu", force)))

    wrapped = call_queue.wrap_gradio_call_no_job(lambda: ("image", "info"), add_stats=False)

    assert wrapped() == ("image", "info")
    assert calls == ["torch", ("cpu", False)]


def test_gradio_call_runs_cpu_gc_after_exception_fallback(monkeypatch):
    calls = []
    disable_memmon(monkeypatch)
    monkeypatch.setattr(call_queue.devices, "torch_gc", lambda: calls.append("torch"))
    monkeypatch.setattr(call_queue.devices, "cpu_gc", lambda force=False: calls.append(("cpu", force)))
    monkeypatch.setattr(call_queue.errors, "report", lambda *_args, **_kwargs: None)

    def raises():
        raise RuntimeError("boom")

    wrapped = call_queue.wrap_gradio_call_no_job(raises, extra_outputs=[None, ""], add_stats=False)

    result = wrapped()

    assert result[0:2] == (None, "")
    assert "RuntimeError" in result[-1]
    assert calls == ["torch", ("cpu", False)]
