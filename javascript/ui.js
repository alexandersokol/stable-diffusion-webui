// various functions for interaction with ui.py not large enough to warrant putting them in separate files

function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function all_gallery_buttons() {
    var allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
    var visibleGalleryButtons = [];
    allGalleryButtons.forEach(function(elem) {
        if (elem.parentElement.offsetParent) {
            visibleGalleryButtons.push(elem);
        }
    });
    return visibleGalleryButtons;
}

function selected_gallery_button() {
    return all_gallery_buttons().find(elem => elem.classList.contains('selected')) ?? null;
}

function selected_gallery_index() {
    return all_gallery_buttons().findIndex(elem => elem.classList.contains('selected'));
}

function gallery_container_buttons(gallery_container) {
    return gradioApp().querySelectorAll(`#${gallery_container} .thumbnail-item.thumbnail-small`);
}

function selected_gallery_index_id(gallery_container) {
    return Array.from(gallery_container_buttons(gallery_container)).findIndex(elem => elem.classList.contains('selected'));
}

function extract_image_from_gallery(gallery) {
    if (gallery.length == 0) {
        return [null];
    }
    if (gallery.length == 1) {
        return [gallery[0]];
    }

    var index = selected_gallery_index();

    if (index < 0 || index >= gallery.length) {
        // Use the first image in the gallery as the default
        index = 0;
    }

    return [gallery[index]];
}

window.args_to_array = Array.from; // Compatibility with e.g. extensions that may expect this to be around

function switch_to_txt2img() {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[0].click();

    return Array.from(arguments);
}

function switch_to_img2img_tab(no) {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}
function switch_to_img2img() {
    switch_to_img2img_tab(0);
    return Array.from(arguments);
}

function switch_to_sketch() {
    switch_to_img2img_tab(1);
    return Array.from(arguments);
}

function switch_to_inpaint() {
    switch_to_img2img_tab(2);
    return Array.from(arguments);
}

function switch_to_inpaint_sketch() {
    switch_to_img2img_tab(3);
    return Array.from(arguments);
}

function switch_to_extras() {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[2].click();

    return Array.from(arguments);
}

function get_tab_index(tabId) {
    let buttons = gradioApp().getElementById(tabId).querySelector('div').querySelectorAll('button');
    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].classList.contains('selected')) {
            return i;
        }
    }
    return 0;
}

function create_tab_index_args(tabId, args) {
    var res = Array.from(args);
    res[0] = get_tab_index(tabId);
    return res;
}

function get_img2img_tab_index() {
    let res = Array.from(arguments);
    res.splice(-2);
    res[0] = get_tab_index('mode_img2img');
    return res;
}

function create_submit_args(args) {
    var res = Array.from(args);

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is sending outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
    // If gradio at some point stops sending outputs, this may break something
    if (Array.isArray(res[res.length - 3])) {
        res[res.length - 3] = null;
    }

    return res;
}

function setSubmitButtonsVisibility(tabname, showInterrupt, showSkip, showInterrupting) {
    gradioApp().getElementById(tabname + '_interrupt').style.display = showInterrupt ? "block" : "none";
    gradioApp().getElementById(tabname + '_skip').style.display = showSkip ? "block" : "none";
    gradioApp().getElementById(tabname + '_interrupting').style.display = showInterrupting ? "block" : "none";
}

function showSubmitButtons(tabname, show) {
    setSubmitButtonsVisibility(tabname, !show, !show, false);
}

function showSubmitInterruptingPlaceholder(tabname) {
    setSubmitButtonsVisibility(tabname, false, true, true);
}

function submitButtonStateForProgress(res) {
    if (res && res.active && (res.interrupted || res.stopping_generation)) {
        return {showInterrupt: false, showSkip: true, showInterrupting: true};
    }

    if (res && res.active) {
        return {showInterrupt: true, showSkip: true, showInterrupting: false};
    }

    return {showInterrupt: false, showSkip: false, showInterrupting: false};
}

function syncSubmitButtonsFromProgress(tabname, res) {
    if (!res || !res.active) return;

    var state = submitButtonStateForProgress(res);
    setSubmitButtonsVisibility(tabname, state.showInterrupt, state.showSkip, state.showInterrupting);
}

function showRestoreProgressButton(tabname, show) {
    var button = gradioApp().getElementById(tabname + "_restore_progress");
    if (!button) return;
    button.style.setProperty('display', show ? 'flex' : 'none', 'important');
}

function getInputNumberValue(elemId, fallback) {
    var input = gradioApp().querySelector("#" + elemId + " input");
    if (!input) return fallback;

    var value = Number(input.value);
    return Number.isFinite(value) ? value : fallback;
}

function getInputCheckboxValue(elemId, fallback) {
    var input = gradioApp().querySelector("#" + elemId + " input[type=checkbox]");
    if (!input) return fallback;

    return input.checked;
}

function estimateImageStorageBytes(width, height, imageFormat) {
    var bytesPerPixel = {
        jpg: 1.0,
        jpeg: 1.0,
        webp: 1.2,
        avif: 1.0,
        png: 2.5
    };

    var format = (imageFormat || "png").toLowerCase();
    return Math.max(1, width) * Math.max(1, height) * (bytesPerPixel[format] || bytesPerPixel.png);
}

function formatStorageBytes(bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        return (bytes / (1024 * 1024 * 1024)).toFixed(1) + " GB";
    }

    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function getTxt2imgStorageDimensions() {
    var width = getInputNumberValue("txt2img_width", 512);
    var height = getInputNumberValue("txt2img_height", 512);
    var enableHr = getInputCheckboxValue("txt2img_hr", false);

    if (!enableHr) return {width: width, height: height, enableHr: false};

    var hrResizeX = getInputNumberValue("txt2img_hr_resize_x", 0);
    var hrResizeY = getInputNumberValue("txt2img_hr_resize_y", 0);
    var hrScale = getInputNumberValue("txt2img_hr_scale", 2);

    if (hrResizeX > 0 && hrResizeY > 0) {
        width = hrResizeX;
        height = hrResizeY;
    } else if (hrResizeX > 0) {
        height = Math.round(height * hrResizeX / width);
        width = hrResizeX;
    } else if (hrResizeY > 0) {
        width = Math.round(width * hrResizeY / height);
        height = hrResizeY;
    } else {
        width = Math.round(width * hrScale);
        height = Math.round(height * hrScale);
    }

    return {width: width, height: height, enableHr: true};
}

function getGridStorageDimensions(width, height, imageCount, batchSize) {
    var cols = Math.max(1, Math.min(batchSize, imageCount));
    var rows = Math.ceil(imageCount / cols);

    return {width: width * cols, height: height * rows};
}

function hasImg2imgMask() {
    return get_tab_index("mode_img2img") >= 2;
}

function estimateGenerationStorage(tabname) {
    var dimensions = tabname == "txt2img" ? getTxt2imgStorageDimensions() : {
        width: getInputNumberValue("img2img_width", 512),
        height: getInputNumberValue("img2img_height", 512),
        enableHr: false
    };
    var batchCount = Math.max(1, Math.round(getInputNumberValue(tabname + "_batch_count", 1)));
    var batchSize = Math.max(1, Math.round(getInputNumberValue(tabname + "_batch_size", 1)));
    var imageCount = batchCount * batchSize;
    var sampleBytes = estimateImageStorageBytes(dimensions.width, dimensions.height, opts.samples_format);
    var totalBytes = 0;
    var imageFiles = 0;
    var textFiles = 0;

    var addImages = function(count, bytesPerImage) {
        if (count <= 0) return;

        imageFiles += count;
        totalBytes += count * (bytesPerImage || sampleBytes);
        if (opts.save_txt) {
            textFiles += count;
            totalBytes += count * 4096;
        }
    };

    if (opts.samples_save) {
        addImages(imageCount);

        if (opts.save_images_before_face_restoration) {
            addImages(imageCount);
        }

        if (tabname == "txt2img" && dimensions.enableHr && opts.save_images_before_highres_fix) {
            addImages(imageCount);
        }

        if (tabname == "img2img" && opts.save_images_before_color_correction) {
            addImages(imageCount);
        }

        if (tabname == "img2img" && hasImg2imgMask()) {
            if (opts.save_mask) addImages(imageCount);
            if (opts.save_mask_composite) addImages(imageCount);
        }
    }

    if (tabname == "img2img" && opts.save_init_img) {
        addImages(1);
    }

    var shouldSaveGrid = opts.grid_save && imageCount > 0 && (imageCount > 1 || !opts.grid_only_if_multiple);
    if (shouldSaveGrid) {
        var gridDimensions = getGridStorageDimensions(dimensions.width, dimensions.height, imageCount, batchSize);
        addImages(1, estimateImageStorageBytes(gridDimensions.width, gridDimensions.height, opts.grid_format));
    }

    return {
        bytes: totalBytes,
        imageFiles: imageFiles,
        textFiles: textFiles,
        imageCount: imageCount,
        width: dimensions.width,
        height: dimensions.height
    };
}

function confirmStorageWarning(tabname) {
    if (!opts.storage_warning_enabled) return true;

    var thresholdMb = Number(opts.storage_warning_threshold_mb || 0);
    if (!Number.isFinite(thresholdMb) || thresholdMb <= 0) return true;

    var estimate = estimateGenerationStorage(tabname);
    var thresholdBytes = thresholdMb * 1024 * 1024;
    if (estimate.bytes < thresholdBytes) return true;

    return confirm(
        "This generation may write about " + formatStorageBytes(estimate.bytes) + " to disk.\n\n" +
        "Images/files: " + estimate.imageFiles + (estimate.textFiles ? " images + " + estimate.textFiles + " text files" : " images") + "\n" +
        "Output size: " + estimate.width + "x" + estimate.height + ", batch images: " + estimate.imageCount + "\n\n" +
        "Continue?"
    );
}

function cancelGenerationSubmit(message) {
    throw new Error(message || "Generation cancelled.");
}

function submitCore(args, warnStorage) {
    if (warnStorage && !confirmStorageWarning('txt2img')) {
        cancelGenerationSubmit("Generation cancelled by storage warning.");
    }

    showSubmitButtons('txt2img', false);

    var id = randomId("txt2img");
    localSet("txt2img_task_id", id);

    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
        showSubmitButtons('txt2img', true);
        localRemove("txt2img_task_id");
        showRestoreProgressButton('txt2img', false);
    }, function(res) {
        syncSubmitButtonsFromProgress('txt2img', res);
    });

    var res = create_submit_args(args);

    res[0] = id;

    return res;
}

function submit() {
    return submitCore(arguments, true);
}

function submit_txt2img_upscale() {
    var res = submitCore(arguments, false);

    res[2] = selected_gallery_index();

    return res;
}

function submit_img2img() {
    if (!confirmStorageWarning('img2img')) {
        cancelGenerationSubmit("Generation cancelled by storage warning.");
    }

    showSubmitButtons('img2img', false);

    var id = randomId("img2img");
    localSet("img2img_task_id", id);

    requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function() {
        showSubmitButtons('img2img', true);
        localRemove("img2img_task_id");
        showRestoreProgressButton('img2img', false);
    }, function(res) {
        syncSubmitButtonsFromProgress('img2img', res);
    });

    var res = create_submit_args(arguments);

    res[0] = id;
    res[1] = get_tab_index('mode_img2img');

    return res;
}

function submit_extras() {
    showSubmitButtons('extras', false);

    var id = randomId("extras");

    requestProgress(id, gradioApp().getElementById('extras_gallery_container'), gradioApp().getElementById('extras_gallery'), function() {
        showSubmitButtons('extras', true);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    console.log(res);
    return res;
}

function attachProgress(tabname, id, options) {
    if (!id) return;

    options = options || {};
    var restoreOnEnd = options.restoreOnEnd !== false;

    localSet(tabname + "_task_id", id);
    showRestoreProgressButton(tabname, false);
    showSubmitButtons(tabname, false);

    requestProgress(id, gradioApp().getElementById(tabname + '_gallery_container'), gradioApp().getElementById(tabname + '_gallery'), function() {
        showSubmitButtons(tabname, true);
        if (restoreOnEnd) {
            showRestoreProgressButton(tabname, true);
        } else {
            localRemove(tabname + "_task_id");
            showRestoreProgressButton(tabname, false);
        }
    }, function(res) {
        syncSubmitButtonsFromProgress(tabname, res);
    }, 0);
}

function restoreProgressTxt2img() {
    var id = localGet("txt2img_task_id");

    localRemove("txt2img_task_id");
    showRestoreProgressButton("txt2img", false);

    return id;
}

function restoreProgressImg2img() {
    var id = localGet("img2img_task_id");

    localRemove("img2img_task_id");
    showRestoreProgressButton("img2img", false);

    return id;
}

function restoreStoredProgress(tabname) {
    var id = localGet(tabname + "_task_id");

    if (!id) return false;

    attachProgress(tabname, id);
    return true;
}

function isMainGenerationTaskType(taskType) {
    return ["txt2img", "img2img"].includes(taskType);
}

function shouldAttachCurrentServerProgress(res) {
    return !!(res && res.active && res.id_task && isMainGenerationTaskType(res.task_type) && !progressSessions[res.id_task]);
}

function attachCurrentServerProgress(res) {
    if (shouldAttachCurrentServerProgress(res)) {
        attachProgress(res.task_type, res.id_task, {restoreOnEnd: true});
    }
}

var currentServerProgressMonitorTimer = null;

function currentServerProgressMonitorPeriod() {
    return Math.max(opts.live_preview_refresh_period || 1000, 1000);
}

function scheduleCurrentServerProgressMonitor() {
    currentServerProgressMonitorTimer = setTimeout(pollCurrentServerProgress, currentServerProgressMonitorPeriod());
}

function pollCurrentServerProgress() {
    request("./internal/progress", {id_task: null, live_preview: false}, function(res) {
        attachCurrentServerProgress(res);
        scheduleCurrentServerProgressMonitor();
    }, function() {
        scheduleCurrentServerProgressMonitor();
    });
}

function startCurrentServerProgressMonitor() {
    if (currentServerProgressMonitorTimer) return;

    pollCurrentServerProgress();
}


/**
 * Configure the width and height elements on `tabname` to accept
 * pasting of resolutions in the form of "width x height".
 */
function setupResolutionPasting(tabname) {
    var width = gradioApp().querySelector(`#${tabname}_width input[type=number]`);
    var height = gradioApp().querySelector(`#${tabname}_height input[type=number]`);
    for (const el of [width, height]) {
        el.addEventListener('paste', function(event) {
            var pasteData = event.clipboardData.getData('text/plain');
            var parsed = pasteData.match(/^\s*(\d+)\D+(\d+)\s*$/);
            if (parsed) {
                width.value = parsed[1];
                height.value = parsed[2];
                updateInput(width);
                updateInput(height);
                event.preventDefault();
            }
        });
    }
}

onUiLoaded(function() {
    restoreStoredProgress('txt2img');
    restoreStoredProgress('img2img');
    startCurrentServerProgressMonitor();
    setupResolutionPasting('txt2img');
    setupResolutionPasting('img2img');
});


function modelmerger() {
    var id = randomId("modelmerger");
    requestProgress(id, gradioApp().getElementById('modelmerger_results_panel'), null, function() {});

    var res = create_submit_args(arguments);
    res[0] = id;
    return res;
}


function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    var name_ = prompt('Style name:');
    return [name_, prompt_text, negative_prompt_text];
}

function confirm_clear_prompt(prompt, negative_prompt) {
    if (confirm("Delete prompt?")) {
        prompt = "";
        negative_prompt = "";
    }

    return [prompt, negative_prompt];
}


var opts = {};
onAfterUiUpdate(function() {
    if (Object.keys(opts).length != 0) return;

    var json_elem = gradioApp().getElementById('settings_json');
    if (json_elem == null) return;

    var textarea = json_elem.querySelector('textarea');
    var jsdata = textarea.value;
    opts = JSON.parse(jsdata);

    executeCallbacks(optionsAvailableCallbacks); /*global optionsAvailableCallbacks*/
    executeCallbacks(optionsChangedCallbacks); /*global optionsChangedCallbacks*/

    Object.defineProperty(textarea, 'value', {
        set: function(newValue) {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            var oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value);
            }

            executeCallbacks(optionsChangedCallbacks);
        },
        get: function() {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            return valueProp.get.call(textarea);
        }
    });

    json_elem.parentElement.style.display = "none";
});

onOptionsChanged(function() {
    var elem = gradioApp().getElementById('sd_checkpoint_hash');
    var sd_checkpoint_hash = opts.sd_checkpoint_hash || "";
    var shorthash = sd_checkpoint_hash.substring(0, 10);

    if (elem && elem.textContent != shorthash) {
        elem.textContent = shorthash;
        elem.title = sd_checkpoint_hash;
        elem.href = "https://google.com/search?q=" + sd_checkpoint_hash;
    }
});

let txt2img_textarea, img2img_textarea = undefined;

function restart_reload() {
    document.body.style.backgroundColor = "var(--background-fill-primary)";
    document.body.innerHTML = '<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';
    var requestPing = function() {
        requestGet("./internal/ping", {}, function(data) {
            location.reload();
        }, function() {
            setTimeout(requestPing, 500);
        });
    };

    setTimeout(requestPing, 2000);

    return [];
}

// Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
// will only visible on web page and not sent to python.
function updateInput(target) {
    let e = new Event("input", {bubbles: true});
    Object.defineProperty(e, "target", {value: target});
    target.dispatchEvent(e);
}


var desiredCheckpointName = null;
function selectCheckpoint(name) {
    desiredCheckpointName = name;
    gradioApp().getElementById('change_checkpoint').click();
}

function currentImg2imgSourceResolution(w, h, scaleBy) {
    var img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] img');
    return img ? [img.naturalWidth, img.naturalHeight, scaleBy] : [0, 0, scaleBy];
}

function updateImg2imgResizeToTextAfterChangingImage() {
    // At the time this is called from gradio, the image has no yet been replaced.
    // There may be a better solution, but this is simple and straightforward so I'm going with it.

    setTimeout(function() {
        gradioApp().getElementById('img2img_update_resize_to').click();
    }, 500);

    return [];

}



function setRandomSeed(elem_id) {
    var input = gradioApp().querySelector("#" + elem_id + " input");
    if (!input) return [];

    input.value = "-1";
    updateInput(input);
    return [];
}

function switchWidthHeight(tabname) {
    var width = gradioApp().querySelector("#" + tabname + "_width input[type=number]");
    var height = gradioApp().querySelector("#" + tabname + "_height input[type=number]");
    if (!width || !height) return [];

    var tmp = width.value;
    width.value = height.value;
    height.value = tmp;

    updateInput(width);
    updateInput(height);
    return [];
}


var onEditTimers = {};

// calls func after afterMs milliseconds has passed since the input elem has been edited by user
function onEdit(editId, elem, afterMs, func) {
    var edited = function() {
        var existingTimer = onEditTimers[editId];
        if (existingTimer) clearTimeout(existingTimer);

        onEditTimers[editId] = setTimeout(func, afterMs);
    };

    elem.addEventListener("input", edited);

    return edited;
}
