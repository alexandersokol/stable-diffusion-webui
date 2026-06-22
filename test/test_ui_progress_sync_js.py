import subprocess
import textwrap


def run_node_script(script):
    result = subprocess.run(
        ["node", "-e", script],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_ui_progress_sync_helpers_attach_only_main_generation_tasks():
    run_node_script(textwrap.dedent(
        r"""
        const fs = require("fs");
        const vm = require("vm");

        const fixedElement = {
            style: {
                display: "",
                setProperty: function(name, value) {
                    this[name] = value;
                },
            },
            parentElement: {style: {display: ""}},
            parentNode: {
                insertBefore: function() {},
                removeChild: function() {},
            },
            querySelector: function() { return null; },
            querySelectorAll: function() { return []; },
            firstElementChild: null,
        };

        const context = {
            console: console,
            window: {location: {href: "http://localhost/"}},
            document: {title: "Stable Diffusion", hidden: false, body: {style: {}, innerHTML: ""}},
            navigator: {},
            opts: {},
            optionsAvailableCallbacks: [],
            optionsChangedCallbacks: [],
            onUiLoaded: function() {},
            onAfterUiUpdate: function() {},
            onOptionsChanged: function() {},
            executeCallbacks: function() {},
            gradioApp: function() {
                return {
                    getElementById: function() { return fixedElement; },
                    querySelector: function() { return fixedElement; },
                    querySelectorAll: function() { return []; },
                };
            },
            localSet: function() {},
            localGet: function() { return null; },
            localRemove: function() {},
            requestGet: function() {},
            confirm: function() { return true; },
            setTimeout: setTimeout,
            clearTimeout: clearTimeout,
            Event: function() {},
            Error: Error,
            XMLHttpRequest: function() {},
            URL: {
                createObjectURL: function() { return "blob:preview"; },
                revokeObjectURL: function() {},
            },
        };
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(fs.readFileSync("javascript/progressbar.js", "utf8"), context);
        vm.runInContext(fs.readFileSync("javascript/ui.js", "utf8"), context);

        function assert(condition, message) {
            if (!condition) throw new Error(message);
        }

        const activeTxt2img = {active: true, id_task: "task(txt2img-AAAAAAA)", task_type: "txt2img"};
        const activeImg2img = {active: true, id_task: "task(img2img-BBBBBBB)", task_type: "img2img"};
        const activeExtras = {active: true, id_task: "task(extras-CCCCCCC)", task_type: "extras"};

        assert(context.shouldAttachCurrentServerProgress(activeTxt2img) === true, "txt2img active task should attach");
        assert(context.shouldAttachCurrentServerProgress(activeImg2img) === true, "img2img active task should attach");
        assert(context.shouldAttachCurrentServerProgress(activeExtras) === false, "extras task should not attach");
        assert(context.shouldAttachCurrentServerProgress({active: false, id_task: activeTxt2img.id_task, task_type: "txt2img"}) === false, "inactive task should not attach");

        context.progressSessions[activeTxt2img.id_task] = {};
        assert(context.shouldAttachCurrentServerProgress(activeTxt2img) === false, "already attached task should not attach again");
        """
    ))


def test_ui_progress_sync_helpers_map_interruption_to_button_state():
    run_node_script(textwrap.dedent(
        r"""
        const fs = require("fs");
        const vm = require("vm");

        const context = {
            console: console,
            window: {location: {href: "http://localhost/"}},
            document: {title: "Stable Diffusion", hidden: false, body: {style: {}, innerHTML: ""}},
            navigator: {},
            opts: {},
            optionsAvailableCallbacks: [],
            optionsChangedCallbacks: [],
            onUiLoaded: function() {},
            onAfterUiUpdate: function() {},
            onOptionsChanged: function() {},
            executeCallbacks: function() {},
            gradioApp: function() { return {getElementById: function() { return null; }, querySelector: function() { return null; }, querySelectorAll: function() { return []; }}; },
            localSet: function() {},
            localGet: function() { return null; },
            localRemove: function() {},
            requestGet: function() {},
            confirm: function() { return true; },
            setTimeout: setTimeout,
            clearTimeout: clearTimeout,
            Event: function() {},
            Error: Error,
            XMLHttpRequest: function() {},
            URL: {createObjectURL: function() {}, revokeObjectURL: function() {}},
        };
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(fs.readFileSync("javascript/progressbar.js", "utf8"), context);
        vm.runInContext(fs.readFileSync("javascript/ui.js", "utf8"), context);

        function assertState(actual, expected, message) {
            const actualJson = JSON.stringify(actual);
            const expectedJson = JSON.stringify(expected);
            if (actualJson !== expectedJson) throw new Error(message + ": " + actualJson);
        }

        assertState(
            context.submitButtonStateForProgress({active: true, interrupted: false, stopping_generation: false}),
            {showInterrupt: true, showSkip: true, showInterrupting: false},
            "active sampling should show interrupt and skip"
        );
        assertState(
            context.submitButtonStateForProgress({active: true, interrupted: true, stopping_generation: false}),
            {showInterrupt: false, showSkip: true, showInterrupting: true},
            "interrupted task should show interrupting placeholder"
        );
        assertState(
            context.submitButtonStateForProgress({active: true, interrupted: false, stopping_generation: true}),
            {showInterrupt: false, showSkip: true, showInterrupting: true},
            "stopping task should show interrupting placeholder"
        );
        """
    ))
