# What the generator needs to become a tool — an audited work list

**What this is.** Every change required to turn the DWN→RTL generator in this repo into something
another person could use on their own model, with each item marked by *when* it should happen
relative to the MNIST port.

**How it differs from the two docs beside it.** `docs/reusable-generator.md` argues *whether* to
build the tool and scopes it in weeks. `docs/mnist-plan.md` covers what MNIST specifically needs.
This is the union, audited against the code as it stands, with file and line references so nothing
here is a guess. Where the three disagree, this one was checked most recently.

**Identifiers.** `B*`, `F*`, `R*` and `T*` are `docs/mnist-plan.md` §2's; `M1a`–`M1g` are its §3
steps. Items that already have an ID there keep it — this document does not invent a second name
for the same thing. `B4`, and the `V*`/`P*`/`Q*` groups below, are new here because no existing ID
covers them.

**Timing legend.**

| | meaning |
|---|---|
| **BEFORE** | do it before MNIST. Either it is wrong today, or MNIST cannot start without it. |
| **DURING** | MNIST is what surfaces or validates it. Doing it earlier means guessing. |
| **AFTER** | packaging and polish. Real work, but it should follow a generator that has stopped moving. |
| **ANYTIME** | independent of MNIST; costs little and unblocks thinking. |

**The organising principle**, from `docs/mnist-plan.md` §1.3, and worth repeating because it is the
test that decides most of these calls:

> *Would this still be right for a dataset we have not thought of?* Not: *does this work for MNIST?*

---

## 1. Defects in shared code — wrong today, not merely JSC-shaped

These are not generalisations. They are bugs that JSC's particular shape hides, and each fails
*silently* rather than erroring.

| # | Defect | Evidence | Why it is dangerous | When | Effort |
|---|---|---|---|---|---|
| **B1** | Feature count hardcoded in the core emitter: `input_bits = 16 * cfg['thermometer_bits']` | `rtlgen/emit_core.py:97` | Emits a core with the wrong input width for any non-16-feature model. No error — Gate 1 fails confusingly, one level away from the cause. **`rtlgen/emit_encoder.py:41` and `tb/gen_vectors.py:74` already derive it correctly** (`thresholds.shape[0]`), so this is one file disagreeing with the two beside it | **BEFORE** | 30 min |
| **R1** | Area model hardcodes the feature count and self-tests at five classes | `dse/area_model.py:39,168,183,222` | Silently wrong predictions for any other dataset — but nothing in the Gate 1 path reads it, so it blocks *predicting* MNIST area, not measuring it | **DURING**, and only if area is predicted before synthesising | 1 h |
| **R2** | Grid is JSC-shaped throughout — size ladder, `tau` anchors, slugs | `dse/grid.py` | Not a defect in the emitter. It means "run a sweep" is not something a user can do on their own model | **AFTER**, and only if the tool offers sweeps | 2–3 h |

**Only B1 is genuinely BEFORE.** R1 and R2 sit in `dse/`, which no part of emit → Gate 1 →
synthesise touches, so neither can block or corrupt the MNIST bring-up. An earlier draft of this
document marked all three BEFORE; that was wrong.

**B1 deserves fixing on `main` regardless of whether the tool ever happens.** It is a live
correctness bug in a shipped emitter.

---

## 2. Generality gaps — where the flow assumes a dataset

Real work, no current failure. Ordered by whether MNIST is blocked without them.

| # | Gap | Evidence | When | Effort |
|---|---|---|---|---|
| **F1** | **Configurable precision.** `Q3.12` is a module-level constant; `HardwareConfig` carries `word_bits`/`frac_bits` but nothing passes them, and `config.py` *asserts* they equal the constants | `exporter/extract.py:118-119`, `rtlgen/config.py:186-187` | **BEFORE** | 1 day incl. verification |
| **B4** | **A dataset descriptor** so dimensions are data, not code | none exists (`datasets/` absent) | **BEFORE** | half day |
| **B2** | **Board record format fixed at 33 bytes** (32 feature + 1 label) | `harness/uart_loader.v:14,64-65`, `scripts/host.py:58` | **DURING** — only if the board path is in scope | 1 day |
| **B3** | **Vector store sized for JSC** — `DATA_W=256`, `DEPTH=1024` | `harness/vector_store.v:24-27` | **DURING** — same condition | half day |
| **P7** | **Default checkpoint paths name a JSC file** in four scripts | `run_gate1.py:27`, `run_tb.py:25`, `host.py:48`, `verify_phase1.py:118` | **AFTER** — they are defaults with overrides, harmless until packaging | 1 h |

**F1 is the one that decides whether MNIST fits at all.** MNIST pixels are natively 8-bit
integers; a 16-bit signed word with twelve fractional bits is roughly double what is needed, and
`REPORT.md` §7 measured that the encoder is where the area goes. Its own finding — an 11-bit word
gives a 5.80× smaller encoder for 0.142 pp on JSC — is the reason this is a *fit enabler* and not
just tidiness.

**On `LABEL_W`:** the harness already parameterises it (`benchmark_fsm.v:35`,
`dwn_basys3_top.v:47`, default 3). MNIST's ten classes need 4 bits, which is a parameter change
rather than a rewrite. Good news, and worth recording so nobody re-audits it.

---

## 3. What only MNIST can settle — **DURING**

These cannot be decided by inspection. Attempting them earlier means guessing.

| Question | Why it needs a second dataset |
|---|---|
| Does the generator emit correct RTL at 784 features and 10 classes? | Every emitted design so far is 16 features / 5 classes. Gate 1 on an MNIST model is the actual test of generality — the whole point of the exercise |
| How many thermometer thresholds per pixel? | `784 × z` is the entire area problem. JSC used 200; MNIST almost certainly cannot afford it. Only measurement says what it can |
| Are MNIST pixels standard-scaled or min-max? | Decides how many integer bits the word needs — the input to F1's derivation logic |
| Does the upstream MNIST recipe binarise differently? | `docs/checkpoint-format.md` is verified against **JSC checkpoints only**. A mismatch produces a valid-looking wrong export, which is the expensive failure mode |
| Does the paper's `1000, 500` fit at any precision? | A negative answer is a publishable result, not a failure (`mnist-plan.md` §3) |

**The most valuable output of MNIST is a list of bugs**, not an accuracy number. Phase 2 and 3
found four JSC-shaped assumptions nobody had noticed by inspection — the `np.packbits` shift at
n<3, the `tau` schedule, the encoder-narrowing fit, the constant selection ratio. A second dataset
is how the next one gets found.

---

## 4. The verification story — what makes the tool worth using

This is the strongest differentiator over rolling your own, and most of it already exists.

| # | Item | State | When |
|---|---|---|---|
| **V1** | **Ship Gate 1 with the tool** | `tb/dwn_core_tb.v` already self-checks and prints `PASS (bit-exact on every vector)` | **ANYTIME** (decision), **AFTER** (packaging) |
| **V2** | **Vectors without a dataset** — generate random inputs, run both golden model and RTL | not built. `tb/gen_vectors.py` assumes a saved `_testvectors.npz` | **AFTER** |
| **V3** | **Simulator independence** — Verilator or Icarus alongside `xsim` | `run_gate1.py` hardcodes Vivado's `xsim` (`:74,115`), but it is isolated behind `find_vivado_bin()`/`run_xsim()`, so it is a backend swap | **AFTER** |
| **V4** | **A precision-choice procedure**, not a constant | `exporter/extract.py` already has `fits_in_word()` and `saturation_is_lossless()`; `experiments/analyze_precision.py` measures the requirement | **DURING** — MNIST is its first real exercise |

**On V1 — ship it.** A generator whose output nobody can check is worth much less, and this
project has the evidence: the emitter's own read-back check reported 20/20 correct while the design
was wrong on 958 of 1,504 vectors. Only an independent golden model caught it.

**On V2 — random vectors are *better* than dataset vectors** for this purpose. You are verifying
the emitter, not the model, and random inputs hit address patterns a trained model's data may never
produce. `experiments/make_test_checkpoint.py` already uses exactly this argument to run Gate 1 at
n=2 and n=4 without a trained model. It also makes `verify()` fully self-contained.

**On V4 — this is the design question the tool turns on.** Precision splits cleanly:

- **integer bits** — derivable exactly from the checkpoint's thresholds, and the *renormalisation*
  trick in `REPORT.md` §7 removes the question entirely: map each feature affinely into [−1, 1) and
  the integer width is 1 for every dataset, forever, with no retraining because a comparison is
  unchanged by a monotonic rescale applied to both sides.
- **fractional bits** — **not** derivable from the checkpoint. Whether quantisation changes
  predictions depends on the data. `REPORT.md` §5.6's scar applies directly: the encoder-narrowing
  result was fitted and validated on the same 1,000 samples, and 8 of 15 features were narrowed too
  far. A tool that picks fractional width from a checkpoint-only heuristic reproduces that bug for
  every user.

So the honest contract: **derive a floor from the checkpoint and say it is a floor; upgrade to
"measured" only when given data.** Never silently claim a width is safe.

---

## 5. Packaging — **AFTER**

Only start once the generator has stopped changing. `docs/reusable-generator.md` §5 gives the
reason: maintaining a fork while the emitters move means merging every change twice.

| # | Item | Notes | Effort |
|---|---|---|---|
| **P1** | Fork `main`, then prune | Fork rather than fresh repo — the history is where the reasoning lives (address bit order, the `__dummy_mapping` trap, the packbits shift) | 1 day |
| **P2** | `pyproject.toml`, one CLI entry point | none exists today | 1 day |
| **P3** | **A licence** | **none exists.** Blocks anyone using it, and blocks citing it | 1 h |
| **P4** | README, worked examples | `rtl/example-model-1x50/` is already a good artifact to build on | 1–2 days |
| **P5** | CI on a second dataset | The claim is generality; it needs a second dataset to be a claim at all | 2–3 days |
| **P6** | Decide the upstream-version policy | Currently one pinned commit (`9f887a0`). Drift fails silently — `__dummy_mapping` has the same shape and dtype as a real mapping | design call |

---

## 6. Decisions to settle early — **ANYTIME**, and they change the scope

Free to answer, and each one narrows what the work above actually is.

| # | Decision | Why it matters now |
|---|---|---|
| **Q1** | **Generator-only, or generator + board flow?** | Decides whether **B2 and B3 exist at all**. Generator-only makes MNIST *Gate 1 only* — roughly 1–2 days instead of the harness rework the Phase 1 ledger scoped. hls4ml, the obvious model, ships no board flow |
| **Q2** | **Does Gate 1 ship?** (V1) | Yes, in my view — and it settles Q1, because Gate-1-only scope means the harness is optional |
| **Q3** | **Which upstream DWN versions?** (P6) | One pinned commit is honest and cheap; a range needs a compatibility layer plus silent-drift detection |
| **Q4** | **Name, and where it lives** | `mnist-plan.md` §1.6 forbids branch- or person-named files; the same discipline should apply to the fork |

**Q1 is the highest-leverage question in this document.** Answering "generator-only" removes B2,
B3 and most of the MNIST harness work in one stroke, and matches how the comparable tools ship.

---

## 7. Explicitly out of scope

Recorded so they are not silently reconsidered:

- **Adopting the encoder narrowing for JSC.** `REPORT.md` §7 argues against it — the binding
  constraint on this device is timing, and the encoder is not on the critical path. F1 makes it
  *possible*; it should not become the default.
- **Learnable Reduction.** Reopened in `docs/phase2-ledger.md` at ~35% of the headline design, but
  never explored enough to belong in a tool. It is a research axis, not a packaging task.
- **Reorganising the JSC artifacts.** `mnist-plan.md` §1.4 — `REPORT.md` and the `jsc-complete`
  tag reference current paths.

---

## 8. Suggested order

```
BEFORE    B1  fix the hardcoded feature count          ← wrong today; fix on main
          Q1  decide generator-only vs board flow      ← free, and removes work
          B4  dataset descriptors
          F1  configurable precision                   ← the fit enabler
                    ↓  JSC must reproduce exactly after each (mnist-plan §1.2)
DURING    M1–M5, V4                                    ← MNIST finds what inspection cannot
          B2, B3 only if Q1 said "board flow"
AFTER     V2, V3   self-contained verification
          P1–P6    fork, package, licence, CI
          R1, R2   sweep tooling, if the tool offers sweeps at all
```

**Rough total, excluding MNIST itself:** two to three days before, one to two weeks after —
consistent with `docs/reusable-generator.md` §4's estimate, and still "almost none of it RTL".

---

## 9. Pointers

- `docs/mnist-plan.md` — ground rules for the branch, and the JSC-must-not-break gate
- `docs/mnist-phase1-ledger.md` — the dated log for the port
- `docs/reusable-generator.md` — whether to build the tool at all, and how to split it off
- `REPORT.md` §7 — the precision measurement F1 exists to expose
- `docs/checkpoint-format.md` — what the exporter reads, verified against JSC only (M4)
