# Task 1 Report: Add Helper Tests

## What I implemented

Created `test/test_vertical_outpainting_mk3.py` with the exact helper-loading stubs and test cases from the task brief. The tests define contracts for:

- All-direction preset expansion.
- Single-direction preset filtering.
- Explicit preset selection and no-direction input.
- Centered zoom preserving image size and corner content.

No production scripts were modified.

## What I tested and test results

The requested command was attempted first:

```text
pytest test/test_vertical_outpainting_mk3.py -q
```

Result: the `pytest` command was not found on PATH.

Fallback command:

```text
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk3.py -q
```

Result: exit code 1, `4 failed, 1 warning in 0.24s`. All four tests failed while loading the future script with `FileNotFoundError` for:

```text
D:\stable-diffusion\v1.10.1\scripts\vertical_outpainting_mk3.py
```

The warning was an existing pytest configuration warning: `Unknown config option: base_url`.

## TDD Evidence

### RED

Command run:

```text
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk3.py -q
```

Relevant output:

```text
FFFF                                                                     [100%]
4 failed, 1 warning in 0.24s
FileNotFoundError: [Errno 2] No such file or directory: 'D:\\stable-diffusion\\v1.10.1\\scripts\\vertical_outpainting_mk3.py'
```

This failure is expected because Task 1 creates only the tests and the future implementation script does not yet exist.

## Files changed

- `test/test_vertical_outpainting_mk3.py`
- `.superpowers/sdd/2026-07-30-vertical-outpainting-mk3/task-1-report.md`

## Self-review findings

- Test content matches the task brief's supplied structure and exact values.
- Scope is limited to the new test module and this report.
- `scripts/poor_mans_outpainting.py` was not modified.
- `scripts/outpainting_mk_2.py` was not modified.
- `scripts/vertical_outpainting_mk3.py` remains absent as required for RED.
- `git diff --check` reported no whitespace errors for tracked diff content.

## Any issues or concerns

- `pytest` is not available as a direct command; the repository environment fallback was used successfully.
- The focused tests are intentionally failing until the Mk3 implementation is added in a later task.
