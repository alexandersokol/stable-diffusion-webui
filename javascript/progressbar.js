// code related to showing and updating progressbar shown as the image is being made

function rememberGallerySelection() {

}

function getGallerySelectedIndex() {

}

function request(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    if (errorHandler) errorHandler();
                }
            } else {
                if (errorHandler) errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function pad2(x) {
    return x < 10 ? '0' + x : x;
}

function formatTime(secs) {
    if (secs > 3600) {
        return pad2(Math.floor(secs / 60 / 60)) + ":" + pad2(Math.floor(secs / 60) % 60) + ":" + pad2(Math.floor(secs) % 60);
    } else if (secs > 60) {
        return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
    } else {
        return Math.floor(secs) + "s";
    }
}


var originalAppTitle = undefined;

onUiLoaded(function() {
    originalAppTitle = document.title;
});

function setTitle(progress) {
    var title = originalAppTitle;

    if (opts.show_progress_in_title && progress) {
        title = '[' + progress.trim() + '] ' + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}


function randomId(taskType) {
    var prefix = taskType ? taskType + "-" : "";
    return "task(" + prefix + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + ")";
}

var progressSessions = {};

function progressRefreshPeriod(multiplier) {
    var basePeriod = opts.live_preview_refresh_period || 500;
    if (!document.hidden) return basePeriod;

    return Math.max(basePeriod * multiplier, 2000);
}

function createProgressSubscriber(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout) {
    var parentProgressbar = progressbarContainer.parentNode;
    var divProgress = document.createElement('div');
    divProgress.className = 'progressDiv';
    divProgress.style.display = opts.show_progressbar ? "block" : "none";

    var divInner = document.createElement('div');
    divInner.className = 'progress';

    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    return {
        id_task: id_task,
        dateStart: new Date(),
        wasEverActive: false,
        parentProgressbar: parentProgressbar,
        gallery: gallery,
        atEnd: atEnd || function() {},
        onProgress: onProgress,
        inactivityTimeout: inactivityTimeout,
        divProgress: divProgress,
        divInner: divInner,
        livePreview: null,
        removed: false
    };
}

function removeProgressSubscriber(session, subscriber) {
    if (subscriber.removed) return;

    subscriber.removed = true;

    if (subscriber.divProgress && subscriber.divProgress.parentNode) {
        subscriber.divProgress.parentNode.removeChild(subscriber.divProgress);
    }

    if (subscriber.gallery && subscriber.livePreview && subscriber.livePreview.parentNode) {
        subscriber.livePreview.parentNode.removeChild(subscriber.livePreview);
    }

    subscriber.atEnd();

    session.subscribers = session.subscribers.filter(function(item) {
        return item !== subscriber;
    });

    if (session.subscribers.length == 0) {
        stopProgressSession(session);
    }
}

function removeAllProgressSubscribers(session) {
    Array.from(session.subscribers).forEach(function(subscriber) {
        removeProgressSubscriber(session, subscriber);
    });
}

function createProgressSession(id_task) {
    return {
        id_task: id_task,
        subscribers: [],
        wakeLock: null,
        progressErrors: 0,
        livePreviewErrors: 0,
        progressTimer: null,
        livePreviewTimer: null,
        lastLivePreviewId: 0,
        progressStarted: false,
        livePreviewStarted: false,
        stopped: false
    };
}

async function requestProgressWakeLock(session) {
    if (!opts.prevent_screen_sleep_during_generation || session.wakeLock) return;
    try {
        session.wakeLock = await navigator.wakeLock.request('screen');
    } catch (err) {
        console.error('Wake Lock is not supported.');
    }
}

async function releaseProgressWakeLock(session) {
    if (!opts.prevent_screen_sleep_during_generation || !session.wakeLock) return;
    try {
        await session.wakeLock.release();
        session.wakeLock = null;
    } catch (err) {
        console.error('Wake Lock release failed', err);
    }
}

function stopProgressSession(session) {
    if (session.stopped) return;

    session.stopped = true;

    if (session.progressTimer) clearTimeout(session.progressTimer);
    if (session.livePreviewTimer) clearTimeout(session.livePreviewTimer);

    releaseProgressWakeLock(session);
    setTitle("");
    delete progressSessions[session.id_task];
}

function progressTextFromResponse(res) {
    var progressText = "";

    if (res.progress > 0) {
        progressText = ((res.progress || 0) * 100.0).toFixed(0) + '%';
    }

    if (res.eta) {
        progressText += " ETA: " + formatTime(res.eta);
    }

    return progressText;
}

function updateProgressSubscriber(session, subscriber, res, titleProgressText) {
    if (subscriber.removed) return;

    var divInner = subscriber.divInner;
    var progressText = titleProgressText;

    divInner.style.width = ((res.progress || 0) * 100.0) + '%';
    divInner.style.background = res.progress ? "" : "transparent";

    if (res.textinfo && res.textinfo.indexOf("\n") == -1) {
        progressText = res.textinfo + " " + progressText;
    }

    divInner.textContent = progressText;

    if (res.active) subscriber.wasEverActive = true;

    if (!res.active && subscriber.wasEverActive) {
        removeProgressSubscriber(session, subscriber);
        return;
    }

    var elapsedFromStart = (new Date() - subscriber.dateStart) / 1000;
    if (elapsedFromStart > subscriber.inactivityTimeout && !res.queued && !res.active) {
        removeProgressSubscriber(session, subscriber);
        return;
    }

    if (subscriber.onProgress) {
        subscriber.onProgress(res);
    }
}

function showProgressReconnect(session) {
    session.subscribers.forEach(function(subscriber) {
        if (subscriber.removed) return;
        subscriber.divInner.style.background = "";
        subscriber.divInner.textContent = "Connection lost. Reconnecting...";
    });
}

function updateLivePreviewSubscriber(subscriber, livePreviewData) {
    if (subscriber.removed || !subscriber.gallery || !livePreviewData) return;

    var img = new Image();
    img.onload = function() {
        if (subscriber.removed || !subscriber.gallery) return;

        if (!subscriber.livePreview) {
            subscriber.livePreview = document.createElement('div');
            subscriber.livePreview.className = 'livePreview';
            subscriber.gallery.insertBefore(subscriber.livePreview, subscriber.gallery.firstElementChild);
        }

        subscriber.livePreview.appendChild(img);
        if (subscriber.livePreview.childElementCount > 2) {
            subscriber.livePreview.removeChild(subscriber.livePreview.firstElementChild);
        }
    };
    img.src = livePreviewData;
}

function sessionHasGallery(session) {
    return session.subscribers.some(function(subscriber) {
        return !subscriber.removed && subscriber.gallery;
    });
}

function startProgressSessionPolling(session) {
    if (session.progressStarted) return;

    session.progressStarted = true;

    var pollProgress = function() {
        if (session.stopped) return;

        requestProgressWakeLock(session);
        request("./internal/progress", {id_task: session.id_task, live_preview: false}, function(res) {
            if (session.stopped) return;

            session.progressErrors = 0;

            if (res.completed) {
                removeAllProgressSubscribers(session);
                return;
            }

            var titleProgressText = progressTextFromResponse(res);
            setTitle(titleProgressText);

            Array.from(session.subscribers).forEach(function(subscriber) {
                updateProgressSubscriber(session, subscriber, res, titleProgressText);
            });

            if (session.stopped) return;

            session.progressTimer = setTimeout(pollProgress, progressRefreshPeriod(4));
        }, function() {
            if (session.stopped) return;

            session.progressErrors += 1;
            showProgressReconnect(session);
            session.progressTimer = setTimeout(pollProgress, Math.min(1000 * session.progressErrors, 5000));
        });
    };

    pollProgress();
}

function startLivePreviewSessionPolling(session) {
    if (session.livePreviewStarted || !sessionHasGallery(session)) return;

    session.livePreviewStarted = true;

    var pollLivePreview = function() {
        if (session.stopped) return;

        request("./internal/progress", {id_task: session.id_task, id_live_preview: session.lastLivePreviewId, live_preview: !document.hidden}, function(res) {
            if (session.stopped) return;

            session.livePreviewErrors = 0;
            session.lastLivePreviewId = res.id_live_preview;

            if (res.live_preview) {
                session.subscribers.forEach(function(subscriber) {
                    updateLivePreviewSubscriber(subscriber, res.live_preview);
                });
            }

            session.livePreviewTimer = setTimeout(pollLivePreview, progressRefreshPeriod(8));
        }, function() {
            if (session.stopped) return;

            session.livePreviewErrors += 1;
            session.livePreviewTimer = setTimeout(pollLivePreview, Math.min(1000 * session.livePreviewErrors, 5000));
        });
    };

    pollLivePreview();
}

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and
// preview inside gallery element. Cleans up all created stuff when the task is over and calls atEnd.
// calls onProgress every time there is a progress update
function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40) {
    if (!id_task || !progressbarContainer) return;

    var session = progressSessions[id_task];
    if (!session) {
        session = createProgressSession(id_task);
        progressSessions[id_task] = session;
    }

    var subscriber = createProgressSubscriber(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout);
    session.subscribers.push(subscriber);

    startProgressSessionPolling(session);
    startLivePreviewSessionPolling(session);
}
