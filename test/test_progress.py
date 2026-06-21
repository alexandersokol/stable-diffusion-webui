import threading
import time
from types import SimpleNamespace

from PIL import Image

from modules import progress, shared
from modules.shared_state import State


def reset_progress_state():
    if shared.state is None:
        shared.state = State()
    progress.opts = SimpleNamespace(live_previews_enable=False, live_previews_image_format="png", live_preview_max_size=1024, progress_restore_image_max_size=1024)

    with progress.progress_lock:
        progress.current_task = None
        progress.pending_tasks.clear()
        progress.finished_tasks.clear()
        progress.recorded_results.clear()
        progress.live_preview_cache.clear()


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


def test_progress_active_task_without_start_time_returns_valid_response():
    reset_progress_state()
    task_id = "task(txt2img-AAAAAAA)"

    progress.start_task(task_id)
    shared.state.time_start = None
    shared.state.job_count = 1
    shared.state.job_no = 0
    shared.state.sampling_steps = 1
    shared.state.sampling_step = 0

    active = progress.progressapi(progress.ProgressRequest(id_task=task_id, live_preview=False))

    assert active.active is True
    assert active.progress == 0
    assert active.eta is None
    assert active.id_task == task_id


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


def test_record_results_stores_lightweight_gallery_snapshot():
    reset_progress_state()
    image = Image.new("RGB", (2048, 1024), "red")
    result = ([image], "generation_info", "html_info", "html_log")

    progress.record_results("task(txt2img-AAAAAAA)", result)

    with progress.progress_lock:
        recorded = progress.recorded_results[0][1]

    assert recorded is not result
    assert recorded[0][0] is not image
    assert recorded[0][0].size == (1024, 512)
    assert recorded[1:] == result[1:]


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


def test_resize_live_preview_image_caps_largest_side():
    image = Image.new("RGB", (2048, 1024))

    resized = progress._resize_live_preview_image(image, 512)

    assert resized.size == (512, 256)


def test_encode_live_preview_reuses_cached_preview(monkeypatch):
    reset_progress_state()
    progress.opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="png", live_preview_max_size=64)
    shared.state.job_timestamp = "20260621120000"
    image = Image.new("RGB", (128, 64))
    save_calls = []
    original_save = Image.Image.save

    def save_once(self, *args, **kwargs):
        save_calls.append(self.size)
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save_once)

    first_preview = progress._encode_live_preview(image, 1)
    second_preview = progress._encode_live_preview(image, 1)

    assert first_preview == second_preview
    assert len(save_calls) == 1
    assert save_calls[0] == (64, 32)
