import threading
import time
from types import SimpleNamespace

from modules import progress, shared
from modules.shared_state import State


def reset_progress_state():
    if shared.state is None:
        shared.state = State()
    if progress.opts is None:
        progress.opts = SimpleNamespace(live_previews_enable=False)

    with progress.progress_lock:
        progress.current_task = None
        progress.pending_tasks.clear()
        progress.finished_tasks.clear()
        progress.recorded_results.clear()


def test_progress_task_lifecycle_is_visible_through_locked_snapshots():
    reset_progress_state()

    progress.add_task_to_queue("task(txt2img-AAAAAAA)")
    pending = progress.get_pending_tasks()

    assert pending.size == 1
    assert pending.tasks == ["task(txt2img-AAAAAAA)"]

    progress.start_task("task(txt2img-AAAAAAA)")
    shared.state.time_start = time.time()
    shared.state.job_count = 1
    shared.state.job_no = 0
    shared.state.sampling_steps = 1
    shared.state.sampling_step = 0
    active = progress.progressapi(progress.ProgressRequest(id_task="task(txt2img-AAAAAAA)", live_preview=False))

    assert progress.get_current_task() == "task(txt2img-AAAAAAA)"
    assert active.active is True
    assert active.queued is False

    progress.finish_task("task(txt2img-AAAAAAA)")
    completed = progress.progressapi(progress.ProgressRequest(id_task="task(txt2img-AAAAAAA)", live_preview=False))

    assert progress.get_current_task() is None
    assert completed.completed is True


def test_record_results_keeps_only_limit():
    reset_progress_state()

    progress.record_results("task(txt2img-AAAAAAA)", "first")
    progress.record_results("task(txt2img-BBBBBBB)", "second")
    progress.record_results("task(txt2img-CCCCCCC)", "third")

    with progress.progress_lock:
        assert progress.recorded_results == [
            ("task(txt2img-BBBBBBB)", "second"),
            ("task(txt2img-CCCCCCC)", "third"),
        ]


def test_restore_progress_waits_until_task_finishes():
    reset_progress_state()
    task_id = "task(txt2img-AAAAAAA)"
    result = ("gallery", "generation_info", "html_info", "html_log")

    progress.add_task_to_queue(task_id)
    progress.start_task(task_id)

    def finish_later():
        time.sleep(0.05)
        progress.record_results(task_id, result)
        progress.finish_task(task_id)

    worker = threading.Thread(target=finish_later)
    worker.start()

    assert progress.restore_progress(task_id) == result
    worker.join(timeout=1)
