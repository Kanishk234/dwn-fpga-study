# Training — how to run

Training runs on Kaggle, not locally. Upstream `torch_dwn` has no CPU path: `lut_layer.py` raises
`EFDFunction CPU not Implemented` in both forward and backward, and only a CUDA extension ships.
See `docs/project-brief.md` §12 risk #7.

Everything else in the project — export, RTL generation, simulation, Vivado — runs locally on CPU.
Only training leaves the machine.

---

## Notebooks

| Notebook | What it does |
|---|---|
| `dwn_jsc_kaggle.ipynb` | Trains one n=6 DWN on JSC, saves a checkpoint + test vectors |

---

## Running it

1. **Create → New Notebook**, then **File → Import Notebook**, upload `dwn_jsc_kaggle.ipynb`.
2. In the settings panel on the right:
   - **Accelerator → GPU** (T4 x2 or P100, either is fine)
   - **Internet → On** — **off by default.** Without it the `git clone` and the OpenML fetch both
     fail, and the errors do not obviously say "turn on the internet."
3. **Run All** (`Ctrl+F9`). Keep the tab open; closing it can kill an interactive session.
4. Download from the **Output** panel (`/kaggle/working/`).

Rough timing: nvcc build 2–5 min, JSC download 1–2 min, training 5–15 min.

---

## What comes out

| File | What it is |
|---|---|
| `dwn_jsc_checkpoint.pt` | config, `state_dict`, **thermometer thresholds**, scaler params, class names, accuracy history |
| `dwn_jsc_testvectors.npz` | 1000 test samples: binarized input, raw input, label, model prediction |

Put both in `training/artifacts/`.

**The thresholds and scaler matter as much as the weights.** `Thermometer` is not an `nn.Module`,
so its thresholds are *not* in `state_dict` — but the hardware encoder is undefined without them.
Same for the `StandardScaler` mean/scale: the thresholds live in scaled space. A checkpoint without
these is not a complete model.

`dwn_jsc_testvectors.npz` is the raw material for the **Gate 1** golden-model testbench — the RTL
has to reproduce that `pred` array exactly.

---

## Changing the configuration

Edit the `CONFIG` dict in one cell. Nothing else should need touching — that's deliberate, and it's
the same rule Phase 2 depends on (`docs/dse-plan.md` §1: a sweep point is a config, not a code
edit). If you find yourself editing code to change a configuration, that's a bug in the notebook.

Default is `t=4`, distributive, layers `[300, 100]`, n=6 — 64 input bits, 400 predicted core LUTs,
~1.9% of an xc7a35t. Deliberately small: near the paper's sm-50 config so there's a reference number
to check against, and safely inside the routing risk (brief §12 risk #2).

Keep `n=6` for Phase 1 (CLAUDE.md). It becomes a sweep axis in Phase 2.

---

## Reading the result

Reference JSC accuracies, from `docs/project-brief.md` §8 (xcvu9p, out-of-context):

| Model | Accuracy | LUTs |
|---|---|---|
| DWN sm-10 | 71.2% | 64 |
| DWN sm-50 | 74.0% | 311 |
| DWN large | 76.3% | 4,972 |
| hls4ml | 76.2% | 63,251 |

The default config is roughly sm-50 scale, so **~74% means the setup is sound.** A couple of points
either way is normal — different split, different seed, no encoder fine-tuning.

**~20% means something is structurally broken**, not badly tuned. That's chance on 5 classes. Look
at the binarization and the `GroupSum` grouping first, not the learning rate.

---

## When it breaks

**`efd_cuda failed to import`** — the CUDA extension didn't compile. Re-run the install cell with
`--quiet` removed and read the nvcc output. Usually a CUDA-version mismatch between the image's
toolkit and the one PyTorch was built against.

**`NameError: efd_cuda` during training** — shouldn't happen; the verification cell exists to catch
this earlier. If it does, the extension didn't build *and* the verification cell was skipped.

**`EFDFunction CPU not Implemented`** — the model or a tensor isn't on the GPU. Check the accelerator
is actually enabled and the session restarted after enabling it.

**Clone or `fetch_openml` hangs or 403s** — internet is off in the settings panel.

**Session dies mid-training** — free-tier quota or the 12h session cap. The checkpoint only gets
written in the last cell, so a death mid-training loses the run. For long sweeps later, save
per-epoch.

---

## Before the Phase 2 sweep

The nvcc build costs 2–5 minutes on every fresh session, which is pure overhead repeated across
dozens of runs. Build the wheel once, save it as a Kaggle Dataset, and install from that instead.

Not worth doing yet — do it when the sweep is actually being set up, not before.
