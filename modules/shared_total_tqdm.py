import tqdm

from modules import console_progress, shared


class TotalTQDM:
    def __init__(self):
        self._tqdm = None

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc=console_progress.generation_progress_desc(shared.state),
            total=shared.state.job_count * shared.state.sampling_steps,
            position=1,
            file=shared.progress_print_out,
            mininterval=console_progress.progress_min_interval(shared.opts),
        )

    def update_description(self):
        if self._tqdm is None:
            return

        desc = console_progress.generation_progress_desc(shared.state)
        if self._tqdm.desc != desc:
            self._tqdm.set_description_str(desc, refresh=False)

    def update(self):
        if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self.update_description()
        self._tqdm.update()

    def updateTotal(self, new_total):
        if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self.update_description()
        self._tqdm.total = new_total

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.refresh()
            self._tqdm.close()
            self._tqdm = None

