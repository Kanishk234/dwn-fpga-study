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
| `dwn_jsc_<run>_checkpoint.pt` | config, `state_dict`, **thermometer thresholds**, scaler params, class names, accuracy + loss history |
| `dwn_jsc_<run>_testvectors.npz` | 1000 test samples: binarized input, raw input, label, model prediction |

Put both in `training/artifacts/`.

`<run>` is `RUN_NAME`, derived from the config: thermometer bits, thermometer kind, layer widths,
and one letter per layer's mapping (`l`=learnable, `r`=random). So `t8_distributive_300-100_lr`.
Runs never overwrite each other, and a checkpoint on disk always names the config that made it.

**Keep losing runs.** Two runs differing in one variable *are* the experiment — the one that scored
worse is the evidence for the one that scored better, and Phase 2's Pareto frontier is built out of
exactly these comparisons.

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

Current config is `t=8`, distributive, layers `[300, 100]`, mapping `['learnable', 'random']`, n=6
— 128 input bits, 400 predicted core LUTs, ~1.9% of an xc7a35t. Deliberately small: near the paper's
sm-50 config so there's a reference number to check against, and safely inside the routing risk
(brief §12 risk #2).

`mapping` takes one entry per layer, `'learnable'` or `'random'`. It used to be a single
`first_layer_mapping` key with `'random'` hardcoded for every later layer — which meant changing
layer 1's wiring required a code edit, in a notebook whose whole premise is that it doesn't.

Use `n=6` for Phase 1 bring-up (CLAUDE.md). It's a real sweep axis in Phase 2, and a config that
fails to route there is a data point, not a mistake.

### Run log

| Run | Encoder bits | Mapping | Final | Best | Notes |
|---|---|---|---|---|---|
| `t4_distributive_300-100_lr` | 4 (64 bits) | learnable, random | 71.95% | 72.31% | Flat from epoch 1; peaked at 17. sm-10's accuracy from ~6× sm-10's nodes. |
| `t8_distributive_300-100_lr` | 8 (128 bits) | learnable, random | 72.52% | 72.53% | **2× the encoder bought +0.22pp best-vs-best.** Encoder is not the binding constraint. Train loss frozen ~0.784 from epoch 7. Batch 256. |
| `t8_distributive_300-100_lr_b32` | 8 (128 bits) | learnable, random | 72.60% | 72.60% | Batch 256 → **32**, matching upstream. **+0.07pp.** Loss still only moves when the LR scheduler fires. |

**Three runs, three rejected hypotheses.** Encoder resolution (+0.22pp for 2× area), batch size
(+0.07pp), and — via the source read — learning rate and layer-1 mapping, both of which turned out to
already match upstream. Every training-recipe knob now matches `examples/mnist.py` exactly, and the
model is still ~1.4pp under sm-50.

The loss signature is identical across all three runs: a slow drift, then a step down exactly when
the LR scheduler fires (epoch 14 and 28), then flat again. It is not the optimizer. **The remaining
difference is architectural** — and reading the paper confirmed it in two places at once:

| | Paper (JSC `sm`, 1× 50) | Ours (all three runs) |
|---|---|---|
| thermometer bits `z` | **200** (3200 input bits) | 4 or 8 (64 / 128 bits) |
| layers | **single** `1× 50` | `[300, 100]`, second layer random-wired |
| tau | 1/0.3 | 1/0.3 ✓ |
| batch size | 100 | 256 / 32 |
| LR schedule | 1e-2(14), 1e-3(14), 1e-4(4) | equivalent ✓ |

| Run | Encoder bits | Layers | Final | Best | Notes |
|---|---|---|---|---|---|
| `t200_distributive_50_l_b100` | 200 (3200 bits) | `[50]` learnable | — | — | **Reproduction attempt. Target 74.0%.** |

Everything in the first three runs was the training recipe, which was never the problem. See
`docs/paper-configs.md`.

`RUN_NAME` gained a `_b<batch>` suffix from this run on — without it this run and the t=8 run
collide on the same name. The two earlier files keep their original names.

**~~What the t4 → t8 comparison settled~~ — RETRACTED.** This previously read: "the encoder-resolution
axis is flatter than the §6 discussion assumes." That conclusion was wrong. The paper uses **z=200**
for every JSC config (`docs/paper-configs.md`); testing 4 vs 8 sampled the flat part of a curve whose
operating point is 25× further out. All three runs below were starved at the input, which is also why
extra layer capacity bought nothing.

**What it did not settle:** why the model is stuck ~1.5pp under sm-50 at comparable node count. The
new evidence is the *training* loss, which barely moves after epoch 7 and only ever drops when the
LR scheduler fires (epoch 14: 0.7845 → 0.7691). Test accuracy tracks it flat. Train loss that won't
fall is underfitting, not a generalization or input-information problem.

Add a row per run. This table is the start of the Phase 2 accuracy axis, and it costs nothing to
keep up to date now versus reconstructing it from checkpoints later.

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

**`Failed building wheel for torch_dwn` / `ModuleNotFoundError: No module named 'torch_dwn'`** —
the wheel build failed, so nothing got installed and the import in the next cell has nothing to
find. Two things to know:

- The install cell must use `pip install --no-build-isolation .`. Upstream's `pyproject.toml`
  lists `torch` as a *build* requirement, so a plain `pip install .` builds in an isolated env
  and pulls a second torch from PyPI — the extension then gets compiled against a torch/CUDA
  pair that isn't the one the session runs on.
- **Never pipe the install through `tail`.** Compiler errors appear near the top of the log; the
  last 20 lines are always the same generic `did not run successfully / See above for output`
  epilogue, which names no cause.

**`efd_cuda failed to import`** — the wheel built but the extension didn't load. Usually a
CUDA-version mismatch between the image's toolkit and the one PyTorch was built against; compare
the two numbers the environment-check cell prints.

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
