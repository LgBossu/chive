# TODO — Runtime environment handling for Torch / XPU subprocesses

## Context (important)

On the development machine, running PyTorch with Intel XPU (SYCL / Level Zero) currently **requires carefully controlled environment variables at process startup**.

This is **not** a Python issue, but a **dynamic linker / runtime selection issue**:

- Torch XPU wheels ship their own SYCL / UR runtime (`libsycl.so`, `libur_loader.so`, adapters)
- The system also provides SYCL / Level Zero via:
  - Ubuntu packages (`libze1`, Mesa stack)
  - Intel GPU repositories (compute runtime, OpenCL, Level Zero)
- If the wrong combination is picked at load time, we see:
  - missing symbols (`urEnqueueCooperativeKernelLaunchExp`)
  - segmentation faults in `libze_loader` or tracing layers
  - crashes during `import torch`

Because of this, **the environment of the process that imports torch matters**.

---

## Current working solution (2026-01)

Torch XPU works reliably *only if the child process is launched with*:

- a **clean `LD_LIBRARY_PATH`**
- an explicit `LD_LIBRARY_PATH` pointing to the venv's `lib/`
- an explicit SYCL backend selection

Example (shell):

```bash
env -u LD_LIBRARY_PATH \
  SYCL_DEVICE_FILTER=level_zero:gpu \
  LD_LIBRARY_PATH="$VENV/lib" \
  "$VENV/bin/python" script.py
```

This ensures:

- Torch uses its bundled SYCL / UR runtime
- System Mesa / Level Zero remain stable for the desktop
- No accidental mixing of runtimes

---

## Implication for this project

Any component that:

- spawns a subprocess
- imports torch inside that subprocess
- runs LLM inference / categorization / embeddings

**MUST pass a controlled environment to the child process.**

Relying on inherited shell state *may be* unsafe.

---

## Python-side solution (recommended)

When spawning subprocesses, explicitly pass `env=`:

```python
env = os.environ.copy()
env.pop("LD_LIBRARY_PATH", None)
env["LD_LIBRARY_PATH"] = "<venv>/lib"
env["SYCL_DEVICE_FILTER"] = "level_zero:gpu"

subprocess.run(
    ["<venv>/bin/python", "worker.py"],
    env=env,
)
```

This guarantees reproducibility regardless of shell, desktop session, or login environment.

---

## TODO — Make this configurable

Later, introduce a **runtime backend config**, e.g.:

```yaml
runtime:
  backend: intel-xpu
  sycl_device: level_zero:gpu
  library_strategy: venv-only
```

or:

```toml
[compute]
backend = "intel-xpu"
sycl_device_filter = "level_zero:gpu"
ld_library_path = "venv"
```

Then:

- map configs → environment variables
- allow switching between:

  - CPU-only
  - Intel XPU
  - CUDA / ROCm (future)
- allow per-machine overrides without code changes

---

## Why this note exists

This setup was non-trivial to debug and **will be forgotten**.
Do not “clean it up” without re-testing XPU imports.

If torch suddenly crashes on import:
→ suspect `LD_LIBRARY_PATH` and SYCL runtime selection first.
