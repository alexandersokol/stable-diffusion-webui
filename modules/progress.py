import base64
import io
import threading
import time

import gradio as gr
from PIL import Image
from pydantic import BaseModel, Field

from modules.shared import opts

import modules.shared as shared
from collections import OrderedDict
import string
import random
from typing import List, Optional

current_task = None
pending_tasks = OrderedDict()
finished_tasks = []
recorded_results = []
recorded_results_limit = 2
progress_lock = threading.RLock()
live_preview_cache = OrderedDict()
live_preview_cache_limit = 8


def get_current_task():
    with progress_lock:
        return current_task


def clear_live_preview_cache():
    with progress_lock:
        live_preview_cache.clear()


def start_task(id_task):
    global current_task

    with progress_lock:
        current_task = id_task
        pending_tasks.pop(id_task, None)
        live_preview_cache.clear()


def finish_task(id_task):
    global current_task

    with progress_lock:
        if current_task == id_task:
            current_task = None

        finished_tasks.append(id_task)
        if len(finished_tasks) > 16:
            finished_tasks.pop(0)
        live_preview_cache.clear()

def create_task_id(task_type):
    N = 7
    res = ''.join(random.choices(string.ascii_uppercase +
    string.digits, k=N))
    return f"task({task_type}-{res})"

def record_results(id_task, res):
    with progress_lock:
        recorded_results.append((id_task, res))
        if len(recorded_results) > recorded_results_limit:
            recorded_results.pop(0)


def add_task_to_queue(id_job):
    with progress_lock:
        pending_tasks[id_job] = time.time()

class PendingTasksResponse(BaseModel):
    size: int = Field(title="Pending task size")
    tasks: List[str] = Field(title="Pending task ids")

class ProgressRequest(BaseModel):
    id_task: Optional[str] = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")
    live_preview: bool = Field(default=True, title="Include live preview", description="boolean flag indicating whether to include the live preview image")


class ProgressResponse(BaseModel):
    id_task: Optional[str] = Field(default=None, title="Task ID")
    task_type: Optional[str] = Field(default=None, title="Task type")
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    completed: bool = Field(title="Whether the task has already finished")
    progress: float = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: float = Field(default=None, title="ETA in secs")
    live_preview: str = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: int = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    textinfo: str = Field(default=None, title="Info text", description="Info text used by WebUI.")


def setup_progress_api(app):
    app.add_api_route("/internal/pending-tasks", get_pending_tasks, methods=["GET"])
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)


def get_pending_tasks():
    with progress_lock:
        pending_tasks_ids = list(pending_tasks)
        pending_len = len(pending_tasks_ids)

    return PendingTasksResponse(size=pending_len, tasks=pending_tasks_ids)


def get_task_type(id_task: Optional[str]):
    if not id_task or not id_task.startswith("task(") or not id_task.endswith(")"):
        return None

    task_name = id_task[5:-1]
    task_type = task_name.split("-", 1)[0]

    return task_type if task_type in {"txt2img", "img2img", "extras"} else None


def _resize_live_preview_image(image, max_size):
    if not max_size or max_size <= 0:
        return image

    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_size:
        return image

    scale = max_size / largest_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resampling_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return image.resize(new_size, resampling_filter)


def _encode_live_preview(image, id_live_preview):
    image_format = opts.live_previews_image_format
    max_size = getattr(opts, "live_preview_max_size", 1024)
    cache_key = (shared.state.job_timestamp, id_live_preview, image_format, max_size)

    with progress_lock:
        cached_preview = live_preview_cache.get(cache_key)
        if cached_preview is not None:
            live_preview_cache.move_to_end(cache_key)
            return cached_preview

    image = _resize_live_preview_image(image, max_size)
    buffered = io.BytesIO()

    if image_format == "png":
        # using optimize for large images takes an enormous amount of time
        if max(*image.size) <= 256:
            save_kwargs = {"optimize": True}
        else:
            save_kwargs = {"optimize": False, "compress_level": 1}
    else:
        save_kwargs = {}

    image.save(buffered, format=image_format, **save_kwargs)
    base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    live_preview = f"data:image/{image_format};base64,{base64_image}"

    with progress_lock:
        live_preview_cache[cache_key] = live_preview
        if len(live_preview_cache) > live_preview_cache_limit:
            live_preview_cache.popitem(last=False)

    return live_preview


def progressapi(req: ProgressRequest):
    with progress_lock:
        current_task_snapshot = current_task
        pending_tasks_snapshot = OrderedDict(pending_tasks)
        finished_tasks_snapshot = set(finished_tasks)

    id_task = req.id_task or current_task_snapshot
    task_type = get_task_type(id_task)

    active = id_task == current_task_snapshot
    queued = id_task in pending_tasks_snapshot
    completed = id_task in finished_tasks_snapshot

    if not active:
        textinfo = "Waiting..."
        if queued:
            sorted_queued = sorted(pending_tasks_snapshot.keys(), key=lambda x: pending_tasks_snapshot[x])
            queue_index = sorted_queued.index(id_task)
            textinfo = "In queue: {}/{}".format(queue_index + 1, len(sorted_queued))
        return ProgressResponse(id_task=id_task, task_type=task_type, active=active, queued=queued, completed=completed, id_live_preview=-1, textinfo=textinfo)

    progress = 0

    job_count, job_no = shared.state.job_count, shared.state.job_no
    sampling_steps, sampling_step = shared.state.sampling_steps, shared.state.sampling_step

    if job_count > 0:
        progress += job_no / job_count
    if sampling_steps > 0 and job_count > 0:
        progress += 1 / job_count * sampling_step / sampling_steps

    progress = min(progress, 1)

    elapsed_since_start = time.time() - shared.state.time_start
    predicted_duration = elapsed_since_start / progress if progress > 0 else None
    eta = predicted_duration - elapsed_since_start if predicted_duration is not None else None

    live_preview = None
    id_live_preview = req.id_live_preview

    if opts.live_previews_enable and req.live_preview:
        shared.state.set_current_image()
        if shared.state.id_live_preview != req.id_live_preview:
            image = shared.state.current_image
            if image is not None:
                live_preview = _encode_live_preview(image, shared.state.id_live_preview)
                id_live_preview = shared.state.id_live_preview

    return ProgressResponse(id_task=id_task, task_type=task_type, active=active, queued=queued, completed=completed, progress=progress, eta=eta, live_preview=live_preview, id_live_preview=id_live_preview, textinfo=shared.state.textinfo)


def restore_progress(id_task):
    while True:
        with progress_lock:
            task_is_pending = id_task == current_task or id_task in pending_tasks

        if not task_is_pending:
            break

        time.sleep(0.1)

    with progress_lock:
        res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)

    if res is not None:
        return res

    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"
