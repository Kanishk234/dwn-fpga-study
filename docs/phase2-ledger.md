# Phase 2 — DSE: running ledger

**Live document. Update it as work lands, not afterwards.** Status table first, then the
chronological log, then the numbers worth quoting, then what is still open.

**Status (2026-08-10): PHASE 2 COMPLETE.** 54 configurations measured, all Gate 1 verified and
placed-and-routed. Headline: **`1x2400 z=50` — the paper's `lg` width — 76.18% at 61.3% of the
device, 101.3 MHz.** Report: `docs/phase2-report.md`. Superseded status note follows.

**Status (2026-08-08): tooling complete, waiting on training.** 2a, 2b and 2d are done and
validated against Phase 1's measured numbers; 2g's tooling is built. **2c is the critical path**
— 32 Kaggle training runs, restarted after the tau fix below. 2e/2f/2g are then one command
each. Phase 1 remains complete (`docs/phase1-report.md`), Gate 1b 166,000/166,000 on hardware.

Phase 2's question (brief §10, Study 1): *what is the accuracy/area/latency Pareto frontier for
DWN on a fixed small FPGA?* Its deliverables are Pareto plots plus one headline number — **the
largest DWN that fits an XC7A35T, and what it scores** — with **core and encoder LUTs reported
separately, always** (brief §6).

**The Phase 1 config is `1x50`** — n=6, z=200, `DistributiveThermometer`, a single
learnable-mapped layer of 50 nodes (the paper's `sm`), 73.84% at 1,619 LUTs. It is the sweep's
first ladder rung and the only config ever run on hardware. Every Phase 2 axis varies from it.

Read first: `docs/dse-plan.md` (what gets swept and why), `docs/phase2-handoff.md` (machine
setup and the restructure decision), `docs/phase1-report.md` (what already works).

---

## Status

| Step | What | Status |
|---|---|---|
| **2a** | **Make the flow config-driven** — see below. Mostly code, not file moves | ✅ baseline reproduces 108/1519/1619 through `dse/run.py` |
| **2b** | Recalibrate the area model against measured encoder cost | ✅ `dse/area_model.py`, 0.5% on the measured config |
| **2c** | Train the Group A grid (GPU-bound, Kaggle, batched) | ✅ 40 models trained across two accounts |
| **2d** | Filter on predicted area before spending any Vivado time | ✅ `grid.should_synthesize()`, with a probe band so the wall is measured |
| **2e** | Synthesize the survivors (serial Vivado, the expensive part) | ✅ 52 configs measured (2 unbuildable, recorded) |
| **2f** | Group B sweeps (pipeline depth, clock, strategy) on survivors only | ✅ 14 variants across 4 rungs |
| **2g** | Merge into one Pareto frontier + the headline number | ✅ **`1x2400 z=50`, 76.18%, 61.30% of device** — edge measured on BOTH area and timing |
| — | *Optional:* n=2 congestion characterization, reported separately | 🟡 n=2 measured as an accuracy/area axis at two rungs (and it lands **on** the frontier). Not characterized for **routing congestion** — both n=2 configs are small and routed cleanly, so the failure mode dse-plan §3 predicts was never reached |

### Prerequisites before any of the above

| Item | Status |
|---|---|
| Machine passes `scripts\verify_phase1.py --with-board` (22/22) | ✅ 2026-08-04, this machine |
| **All sweep synthesis from ONE machine and ONE Vivado version** | decide before 2e |
| Full 166k test set present (gitignored, does not travel) | ✅ here |
| `rtl/gen/` retired in favour of `build/<config>/rtl/` | part of 2a |

---

## Step 2a in detail — what "config-driven" actually means

Roughly 20% file moves, 80% code. `docs/phase2-handoff.md` Part 2 has the directory decision;
this is the work behind it.

**The mechanical part** (minutes): `exporter/emit_core.py` and `emit_encoder.py` move to
`rtlgen/` — brief §11 splits *exporter* (checkpoint → tables/wiring/thresholds) from *rtlgen*
(that export → Verilog), and Phase 2 needs them separable because the filtering step estimates
area from a checkpoint **without** emitting RTL. `rtl/gen/` is deleted; output moves to
`build/<config>/rtl/`. `rtl/` (the hand-written primitives) does not change.

**The actual work** — four things are hardcoded today that a sweep must vary:

| Hardcoded now | Needs to become |
|---|---|
| `emit_core.py` writes to `rtl/gen/dwn_core.v` | an output directory argument — otherwise sweep point #7 overwrites #8 |
| `PIPE_LUT/POP/OUT = 1` as module-level constants | config fields, since Group B sweeps exactly these |
| `run_gate1.py`'s `RTL_SOURCES`, `run_synth.py`'s `TARGETS` | derived from the config's build directory |
| "config" = the `CONFIG` dict inside a checkpoint | a first-class object spanning **training params + hardware params** (pipeline depth, clock target, strategy) — the checkpoint knows nothing about the latter |

Then `dse/` is written on top: the grid, the loop over configs, report parsing into a results
table, Pareto computation, plots. It should **import** `run_xsim()`/`find_vivado_bin()` from
`scripts/run_gate1.py` and `run_one()` from `scripts/run_synth.py` rather than shelling out and
parsing stdout — those were exposed in Phase 1 for this.

**Why this is load-bearing, not tidying:** `dse-plan.md` §1 requires that a sweep point be a
config, not a code edit. If changing configuration means editing a `.py` or `.v`, every sweep
point synthesizes code that has not passed Gate 1 — forty points would mean forty
re-verifications.

**What is already done:** `extract.py` and both emitters handle arbitrary layer counts, widths,
`n` and class counts. What is missing is plumbing: where they write, and what tells them what to
build.

⚠️ **Correction (2026-08-08).** This previously read *"Phase 1 proved it by running the same code
over a `[300,100]` two-layer model"*. Phase 1 ran only `extract.py` over that model — it never
emitted or simulated multi-layer RTL, so the claim overstated the evidence. Multi-layer is now
genuinely verified (Gate 1, 400 nodes, 2 layers — see the log), and `n < 3` turned out to be
**broken** until 2026-08-08, which the original claim would have discouraged anyone from
checking.

## What Phase 1 already answered

`docs/dse-plan.md` §7 listed four open questions. Three are now measured, and two of the answers
change the plan.

### 1. Encoder cost — far worse than budgeted ⚠️

| | |
|---|---|
| anticipated (brief §12 risk #3, Mecik & Kumm) | up to **3.2×** the core |
| **measured** | **14.06×** — 1519 LUTs of encoder against 108 of core |

**The §5 area formula must be rebuilt around this before it is used to filter anything.**
Filtering with a 3.2× assumption would pass configs that cannot fit by a factor of four.

Worse, encoder cost does **not** scale with the core. It scales with the number of *distinct
thresholds the mapping selects*, which grows with node count and **saturates at
`features × z` = 3200 comparators**:

| Config | Nodes | Comparators | Encoder LUTs | Core LUTs | vs 20,800 |
|---|---|---|---|---|---|
| `sm` 1×50 | 50 | **202 (measured)** | **1,519** | 108 | **7.8%** ✅ |
| `md` 1×360 | 360 | ≤ 2,160 | ≲ 16,000 | ~720 | ~~~80% — tight~~ **see below** |
| `lg` 1×2400 | 2400 | → ~3,200 | ~24,000 | ~4,972 | **>100%** ❌ |

⚠️ **`md`'s row is superseded.** It used the *upper bound* of 2,160 comparators (every wiring
slot distinct). With the measured 67% selection ratio it is **1,454 comparators and 58.2% of the
device** — a comfortable rung, not a marginal one. See the 2b log entry.

**`lg` almost certainly does not fit on a Basys 3, and the network is not the reason.** Its 4,972
core LUTs are 24% of the part, exactly as brief §6 predicts. Plan the size ladder against the
encoder, not the core.

*(`md`/`lg` comparator counts are bounds. At `sm` the mapping selected 202 distinct bits from 300
slots — 67% — so real numbers may land lower. Only training those configs settles it.)*

### 2. Pipeline depth does move Fmax — Group B does not collapse

dse-plan §7 asked whether extra stages buy anything. Measured on `dwn_top`:

| Stages | Cycles | LUTs | FF | Fmax |
|---|---|---|---|---|
| 1 | 1 | 1619 | 196 | 84.2 MHz ❌ |
| 2 | 2 | 1619 | 246 | 94.6 MHz ❌ |
| 3 | 3 | 1619 | 249–266 | 115.7–122.9 MHz ✅ |
| **4** | **4** | **1619** | **269** | **161.0 MHz** ✅ |

Nearly 2× across the range, and **LUT count never changes** — pipelining costs flip-flops, not
logic. So Group B is a real axis, it is cheap (no retraining), and 3 stages is the floor for the
board's 100 MHz.

### 3. Reduction cost — the deferred decision now has a number 🟡

dse-plan set the bar explicitly: *"if it's 40% of area, building the pyramid is obviously worth
it; if it's 3%, this was never an interesting axis."*

`dwn_core` is **108 LUTs** for 50 nodes plus 5 × popcount(10) and a 5-way argmax. At one LUT6
per node, the reduction is **~58 LUTs — roughly 54% of the core, and slightly larger than the
network itself.** That matches the paper's warning that in small models "the popcount circuit can
be as large as the network."

⚠️ **This is inference by subtraction, not a measurement.** Vivado inlined `lut_node`, `popcount`
and `argmax` into the top level, so the hierarchical report attributes everything to `dwn_core`.
**Confirm it by synthesizing the reduction standalone before committing to build Learnable
Reduction** — but on this evidence it clears dse-plan's bar comfortably.

### 4. Real widths for the size ladder

`sm` = 1×50 is known-good end to end at **73.83%** (paper: 74.0%). The paper's other JSC points
are `md` = 1×360 (75.6%) and `lg` = 1×2400 (76.3%), all at z=200, single layer, tau tracking
width (1/0.7, 1/0.3, 1/0.1, 1/0.03 for 10/50/360/2400 nodes).

---

## Constraints carried out of Phase 1

Things that will silently produce wrong sweep points if forgotten:

- **Final layer width must be divisible by `num_classes`.** `GroupSum` zero-pads silently
  otherwise, and hardware and software then disagree about group boundaries
  (`docs/checkpoint-format.md` §4). The emitter asserts this; the sweep grid must respect it.
- **`tau` tracks layer width.** It is not a constant to copy from `sm` — see the paper's values
  above.
- **Vector store depth limits batch size**, and bigger models mean wider vectors. `DEPTH=1024` ×
  259 bits ≈ 265 Kbit today, ~15% of block RAM.
- **The Q3.12 reference, not the float model**, is what hardware is scored against. Quantization
  is spec, not error (Phase 1 lost a debugging cycle to this).
- **One machine, one Vivado version**, or the frontier is two half-frontiers.

---

## Log

### 2026-08-04 — Phase 1 reproduced on this machine (the Phase 2 entry gate)

`scripts/verify_phase1.py` exists to answer one question before any sweep starts: *are this
machine's numbers comparable to Phase 1's?* A Pareto frontier assembled from two toolchains is
two half-frontiers with an unknown offset. Run here from a clean tree, no cached `build/`.

**Everything deterministic matched exactly.** Simulation and out-of-context synthesis:
12/12 checks. Then bitstream, program, and Gate 1b across the full test set:

```
Gate 1               : core 1504/1504 · top 1518/1518 · 50 nodes · 202 comparators
harness unit tests   : PASS
dwn_core             : 108 LUT   73 FF
thermometer_encoder  : 1519 LUT   0 FF
dwn_top              : 1619 LUT 269 FF
board design         : 2058 LUT 865 FF 8 BRAM 0 DSP · WNS +1.753 ns
Gate 1b              : 166000/166000 · 166815 core cycles
accuracy             : float32 73.8361% · Q3.12 73.8349%
float-vs-fixed       : 30/166000 (0.0181%) · 1 saturation event
```

`166815 = 166000 + 162 × (LATENCY+1)`, so **II=1 held on silicon again**. Only the stochastic
quantities moved, all within the tolerance the script allows: wall clock 11.4 s (was 11.2),
link rate 14,537/s (was 14,823), I/O wall 6,846× (was 6,713×).

**Python 3.14, not the documented 3.12 — and it was not a confound.** The `.venv` here was
empty and this machine no longer has 3.12 registered (`py --list` shows only 3.14 and 3.11).
Installed the *exact* pins (`numpy==2.3.4`, `torch==2.13.0`, `pyserial==3.5`), which have cp314
wheels, and every bit-exact result held. Reasoning: LUT/FF counts are pure Vivado 2025.2 with no
Python involved, and the golden model's numerics come from numpy's compiled kernels, which do
not change across interpreter minor versions. **Do not read this as permission to drop the 3.12
pin** — `requirements.txt` justifies it on Kaggle parity and on Phase 3's hls4ml/conifer lagging
new Python releases, and neither claim was tested here. It means only that Phase 1's *results*
are interpreter-independent.

**Two corrections came out of this run:**

- `docs/phase1-ledger.md`'s board table read **2054 LUTs / +1.662 ns**; the measured value is
  **2058 / +1.753 ns**, matching the report, handoff, manifest and `verify_phase1.py`. Ledger
  corrected in place with the retraction kept visible.
- The 166k test set is gitignored and **does not travel with the repo**. It was absent here and
  cannot be regenerated locally — upstream `torch_dwn` has no CPU path, so even inference needs
  a GPU. Copied from the Phase 1 machine and verified before use: 166,000 samples, software
  accuracy 73.8361%, first 1000 predictions 1000/1000 against the committed `testvectors.npz`,
  and `x_raw` max diff 4.768e-07 — the recorded 1-ULP `scaler.transform()` fingerprint, which
  is what proves it is the same dump rather than a re-derived one.

**Verdict: this machine is safe to run Phase 2 sweeps on**, and its points are comparable to
Phase 1's. Re-run afterwards as the single scripted command: **22/22 passed.**

### 2026-08-07 — 2a step 1: the config object exists

`rtlgen/config.py`. First piece of 2a, and the first thing to live in `rtlgen/` — which until
now did not exist at all, despite being in the repo layout. **Nothing was moved and no existing
file was touched**, so the flow provably cannot have changed and 22/22 still stands.

`Config = ModelConfig (from the checkpoint) + HardwareConfig (chosen per run)`. The split is the
point: the checkpoint knows only about *training*, while pipeline depth, clock, part and
strategy are Group B axes swept on an already-trained model. Putting them in the checkpoint
would be wrong — one checkpoint feeds several hardware configs. `with_hw()` produces a Group B
variant without retraining, which is exactly what 2f needs.

**Every default reproduces Phase 1 exactly**, so the object can be threaded through the flow
without changing an emitted bit. `python rtlgen/config.py` asserts that against the constants
still owned by other modules — `emit_core.PIPE_LUT/POP/OUT`, `extract.WORD_BITS/FRAC_BITS`,
`run_synth.DEFAULT_PART`, and latency == 4. If someone edits one and not the other it fails
loudly instead of desynchronizing silently. That guard is temporary by design: it deletes itself
when those modules start reading the config instead of owning constants.

Two details worth recording:

- **`name` carries every swept axis** (`n6_z200_distributive_w50_q16.12_p1111_c10`). Two configs
  differing in any axis must get different directories, or one silently overwrites the other's
  build — the exact failure the object exists to prevent.
- **`latency` is derived** (`pipe_enc+lut+pop+out`), never hand-copied. `benchmark_fsm` aligns
  labels with it and a drifted value scores every sample against the wrong answer — a bug that
  has already happened once in this project.
- `ModelConfig.__post_init__` rejects `layers[-1] % num_classes != 0` up front, so a bad sweep
  point dies before Vivado is launched rather than after GroupSum silently zero-pads.

**Next**, in an order that keeps `verify_phase1.py` green after every step: output dir as a
parameter → pipeline depth from config → path derivation in `run_gate1`/`run_synth`/
`build_bitstream` → *then* the moves (`rtlgen/`, `build/<config>/rtl/`, delete `rtl/gen/`).
The moves come last because by then everything already takes paths as inputs.

### 2026-08-07 — 2a step 2: both emitters take an output directory

Smaller than planned, because **`emit_encoder.py` already had `--outdir`**. The real defect was
an asymmetry: `emit_core.py` took `--out <file>` while `emit_encoder.py` took `--outdir <dir>`,
so the two emitters wanted different kinds of argument for the same idea — awkward for a caller
that hands both a single `Config.rtl_dir`. `emit_core.py` now takes `--outdir` too and derives
`dwn_core.v` / `dwn_core_params.vh` from it. Defaults are unchanged (`rtl/gen`).

`--out` is **gone, not deprecated**. Nothing called it — `run_gate1.py` passes no output flag at
all and relies on the default — so keeping both spellings would have been dead weight.

**Verified three ways, not one:**

- emitted into a throwaway `build/configs/` directory: all five files land there, read-back
  checks 50/50 nodes and 202/202 comparators
- regenerated into the default location: **`git diff rtl/gen/` is empty** — byte-identical to
  the committed Phase 1 output, which is the strongest available regression check while
  `rtl/gen/` is still committed
- custom directory vs default: all five files `cmp`-identical
- **Gate 1 re-run: 1504/1504 core, 1518/1518 top, PASS.** Byte-identical output means it could
  not have failed, but the emitter changed and the rule is that confidence is not verification.

`rtlgen/config.py`'s self-test still passes, so the pipeline constants have not drifted.

### 2026-08-10 — THE CORNER RESULT: the paper's `lg` fits, and the wall is timing

Six width×z configs measured. **Every predicted accuracy landed within 0.10 pp** -- the method
(ladder base + measured z-penalty) held.

| config | acc | LUTs | % dev | Fmax | |
|---|---|---|---|---|---|
| `1x800 z=50` | 75.95 | 6,981 | 33.6 | 104.9 | |
| `1x1200 z=50` | 76.05 | 8,444 | 40.6 | 102.3 | |
| `1x1600 z=100` | **76.35** | 13,729 | 66.0 | 101.8 | best accuracy that fits |
| **`1x2400 z=50`** | 76.18 | **12,751** | **61.3** | 101.3 | **new headline** |
| `1x2400 z=100` | 76.39 | 16,681 | 80.2 | 96.2 | ❌ misses the clock |
| `1x3000 z=50` | 76.16 | 13,972 | 67.2 | 96.2 | ❌ misses the clock |

#### ⚠️ Correction to the Phase 1 projection, and it is a large one

`docs/phase1-ledger.md` projected the paper's `lg` (1×2400) as **">100% of the device — does not
fit"**, on an encoder estimate of ~24,000 LUTs against a 20,800-LUT part. That was computed at
z=200 and reported as a property of the *model*.

Measured at z=50, `lg`'s encoder is **5,753 LUTs** and the whole design occupies **61.3%**. The
projection was wrong by roughly 4× on the term that dominated it. **The limit was never the
network — it was paying for z=200**, which this sweep shows buys nothing above z≈50.

#### The binding constraint changed identity

Every config that now fails, fails on **timing with area to spare**:

- `1x3000 z=50` -- 67.2% of the device, **96.2 MHz**
- `1x2400 z=100` -- 80.2% of the device, **96.2 MHz**
- `1x2000` (z=200) -- 102.8%, the one genuine *area* failure

Four pipeline stages is the architectural maximum for a single-layer model, and Group B showed
removing a register makes timing worse, so **neither failure has a remedy in the current RTL**.
The untried candidate is a multi-layer model of the same node count: `2x1200` would gain a fifth
register stage *and* a smaller encoder, since only the first layer reads thermometer bits.

⚠️ Both failures land on **exactly 96.2 MHz**. With two points that is as likely coincidence as
structure; not investigated.

#### The mechanism: the knees compound because they cost in different places

Width buys accuracy and spends **core** LUTs. `z` spends **encoder** LUTs and, past ~50, buys
almost nothing. The encoder is the larger term at every size the paper reports, so cutting `z`
frees budget to buy width. The cleanest single comparison:

**`1x1600 z=100` vs `1x1600` — identical accuracy (76.35%), 13,729 vs 18,777 LUTs.** 27% less
silicon for no accuracy cost at all.

#### A bug the result exposed in `report.py`

The headline picked the largest config by node count among those inside the LUT budget --
**area only**. With `1x3000 z=50` measured (67% of the device, 96.2 MHz) it would have crowned a
design that cannot be clocked on the target board. Now requires `meets_timing` as well. Same
class of error as trusting Vivado's exit status over the measured routed area, which this phase
also hit.

### 2026-08-09 — Group B extended to four rungs, and it overturned the reported result

Group B had run on `1x360` alone -- one measurement, written up as *"no reduced-pipeline variant
meets the board clock."* Slack at 4 stages varies enormously with width, so that was a
one-point claim stated generally. Extended to 50 / 360 / 600 / 1600 (9 new configs, no training,
~5 min each).

| rung | slack @ 4 stages | no OUT | no POP | 2-stage |
|---|---|---|---|---|
| `1x50` | +3.200 ns | **113.9 ✅** | **107.5 ✅** | **101.6 ✅** |
| `1x360` | +0.440 ns | 99.4 ❌ | 72.5 ❌ | 66.5 ❌ |
| `1x600` | +0.400 ns | **102.7 ✅** | 64.5 ❌ | 61.3 ❌ |
| `1x1600` | +0.310 ns | **100.9 ✅** | 58.6 ❌ | 50.7 ❌ |

**The old claim was false at three of the four rungs.** At `1x50` every variant passes --
including 2-stage at **19.7 ns**, half the baseline latency. Corrected in `phase2-report.md`
§4.6 with the retraction kept visible.

**The real rule, and it is clean: drop the output register, never the popcount one.** Removing
the argmax register costs 6 MHz at `1x50` and 2 MHz at `1x1600`; removing the popcount register
costs 6 / 27 / 38 / **42 MHz** as width rises. The popcount is an adder tree whose depth grows
with layer width and is the critical path; the argmax is a 5-way comparison and nearly free.

**It buys real latency where it matters.** `1x1600` -- the largest config that fits -- runs at 3
stages in **29.7 ns against 38.8, a 23% cut, with 15 fewer LUTs**, still meeting 100 MHz.

⚠️ **Two margins are inside placement noise**: `1x1600` passes at **+0.086 ns**, `1x360` fails at
**−0.059 ns**. Phase 1 measured placement varying by tens of picoseconds between runs, so
neither is settled without a repeat, and `1x360` failing while `1x600` passes is two configs
either side of a knife edge rather than a trend.

### 2026-08-09 — Vivado thread count raised, verified not to move the numbers

The whole sweep ran at Vivado's default of **2 threads on a 16-core machine** -- ~12% of
capacity, because `build.tcl` never set `general.maxThreads`. Raising it is a one-line change,
but not a free one: **Vivado's placer and router are multithreaded and the tool documents that
results may differ between thread counts.** This project's comparability argument is that every
sweep point comes from one machine and one flow, so a silent change in placement would split the
frontier exactly the way two Vivado versions would.

Tested rather than assumed. `1x600` re-synthesized at 8 threads:

| | 2 threads | 8 threads |
|---|---|---|
| `dwn_core` LUTs | 1312 | **1312** |
| `thermometer_encoder` LUTs | 9367 | **9367** |
| `dwn_top` LUTs | 10631 | **10631** |
| `dwn_top` WNS | +0.400 | **+0.400** |

**Bit-identical**, so configs measured before and after remain comparable. Now the default in
`build.tcl`, overridable via `DWN_VIVADO_THREADS`. The speedup is unmeasured -- the test
overlapped with another Vivado run, so wall-clock was contaminated; correctness was the point.

### 2026-08-09 — `build/` is now safely disposable

`build/dse/results.json` was the one thing under `build/` that CLAUDE.md's rule does not cover:
everything there is meant to be "regenerable by re-running the flow that made it", and this is
not -- regenerating it costs a Kaggle session plus hours of Vivado. That made `build/` unsafe to
delete, the opposite of the rule's intent.

`load_results()` now falls back to the committed snapshot (`docs/results/sweep-results.json`),
so wiping `build/` costs nothing but disk and the next run rebuilds only what is genuinely
missing. **The ordering matters: snapshot, commit, then delete** -- the fallback recovers what
was committed, not what was not.

### 2026-08-09 — Corner configs queued: the two knees were never combined

One-factor-at-a-time has a structural blind spot. Every non-baseline axis value was only tested
at widths 200 and 360, so **no pair of non-baseline values was ever tried** -- and the sweep
found two independent knees (width saturates ~600, z saturates ~50) without ever asking what
happens when both are taken.

Ranked every pair before choosing, and only one survives:

| pair | verdict |
|---|---|
| width × z | **the only one worth running** -- both terms nearly free |
| width × n | n=2 costs −1.25 pp vs z=50's −0.24; n=4 costs −0.38 for less saving than z=100's −0.10. Dominated |
| width × layers | layer penalty −0.75 pp. Dominated |
| width × encoding | 0.12 pp spread, below the 0.15 pp noise floor |
| z × n | **structurally pointless** — at z≤100 the encoder is already saturated, so cutting n barely shrinks it while still costing accuracy |

Six configs queued for training. The sharpest is **`1x2400 z=100`**: the paper's `lg`, which the
Phase 1 ledger projected as ">100% of the device, does not fit" -- computed at z=200. At z=100 it
is predicted at **84.6%**. If it holds, the paper's largest JSC model fits on a Basys 3 and the
limit was never the network. `1x3000 z=50` is predicted at 71.1% — larger than anything the
paper reports.

### 2026-08-09 — THE WALL, MEASURED: `1x1600` fits, `1x2000` does not

The ladder was extended to 1600 and 2000 because the original sweep found **nothing that failed**
-- the largest config tried used 51% of the device, so brief §10's headline could only be
answered as "the largest we tried". Both trained, both synthesized, and the edge is now measured.

```
HEADLINE: largest DWN measured to fit an XC7A35T
  1x1600 -- 1600 nodes, n=6, z=200
  76.35% accuracy
  18,777 LUTs (90.27% of device) = core 4,124 + encoder 14,316
  0 BRAM, 0 DSP
  103.2 MHz, latency 4 cycles, II=1

THE EDGE:
  1x2000 -- 21,382 LUTs (102.80% of device), WNS -0.538 ns (94.9 MHz)
            fails on BOTH axes: over the part AND misses the board clock
```

#### ⚠️ A config can complete Vivado's flow and still not fit

`1x2000` was recorded `status: ok`, because `run_one` only checks that the flow returned
success. It did:

```
post-synthesis : 20,126 LUTs  (96.76%)   <- fits, so placement proceeded
post-route     : 21,382 LUTs (102.80%)   <- physical optimization pushed it over
place_design completed successfully
```

Placement starts from the post-synthesis netlist, which fit. Physical optimization then
replicated logic chasing timing and pushed the routed design past the device. **"Did it fit" has
to be judged on the measured routed area and on timing, never on whether the tool exited zero.**
`report.py` now does that -- its old caveat keyed off `status == 'synth-failed'` and would have
gone on claiming no edge had been found while the data plainly showed one.

#### The full ladder

| config | acc | core | encoder | top | % dev | Fmax | enc/core |
|---|---|---|---|---|---|---|---|
| 1x50 | 73.84 | 108 | 1,519 | 1,619 | 7.78 | 147.1 | 14.1× |
| 1x100 | 74.81 | 206 | 2,608 | 2,814 | 13.53 | 139.9 | 12.7× |
| 1x200 | 75.32 | 466 | 4,570 | 5,036 | 24.21 | 113.9 | 9.8× |
| 1x360 | 75.85 | 868 | 7,138 | 8,006 | 38.49 | 104.6 | 8.2× |
| 1x500 | 76.04 | 1,118 | 8,432 | 9,542 | 45.88 | 110.6 | 7.5× |
| **1x600** | **76.10** | 1,312 | 9,367 | 10,631 | **51.11** | 104.2 | 7.1× |
| 1x800 | 76.20 | 1,878 | 10,866 | 12,904 | 62.04 | 100.3 | 5.8× |
| 1x1200 | 76.30 | 2,868 | 12,824 | 16,061 | 77.22 | 100.7 | 4.5× |
| **1x1600** | **76.35** | 4,124 | 14,316 | 18,777 | **90.27** | 103.2 | 3.5× |
| 1x2000 | 76.43 | 5,496 | 15,538 | 21,382 | **102.80** ❌ | 94.9 ❌ | 2.8× |

**Zero BRAM and zero DSP at every rung**, and across all 35 measured configs. That column is the
claim against hls4ml (whose MLPs spend DSPs on multiply-accumulate) and conifer, and it is now
measured rather than observed once.

#### The finding that matters more than the headline

**Accuracy saturates around 600 nodes.** From `1x600` to `1x1600` -- a 2.7× increase in nodes and
39 points of device occupancy -- accuracy moves **76.10% → 76.35%, i.e. 0.25 pp**, against a
measured run-to-run noise floor of **0.15 pp**. The last three rungs are within noise of each
other.

So the honest reading of brief §10 is two-sided: the largest that fits is `1x1600` at 90% of the
device, **and it is not meaningfully better than `1x600` at 51%.** A frontier is more useful than
a maximum here.

**`1x1200` also matches the paper's `lg` (1×2400) accuracy of 76.3% at half the width** -- which
the paper could not have seen, because it never swept between `md` and `lg`.

#### The encoder/core ratio inverts across the ladder

**14.1× → 2.8×.** Phase 1's headline "the encoder costs 14× the core" is a small-model artifact,
as suspected -- but the sweep shows it does not merely shrink, it **inverts**: comparators
saturate toward the `features × z` = 3200 ceiling while the core grows at ~1 LUT/node, so past
~1200 nodes the CORE is the growth term. That is the reverse of the entire Phase 1 story, and it
happens exactly where the paper's `lg` config sits.

### 2026-08-09 — THE SWEEP: 35 configs measured, and the area model was rebuilt on them

Ran `dse/run.py --all --impl` over the grid. **35 configs synthesized and placed-and-routed in
~3 hours** -- against a 7 h budget, because the same over-prediction that broke the filter also
made every design smaller and faster to route than planned.

#### Headline

```
1x1200   76.30%   16,061 LUTs (77.22% of device) = core 2,868 + encoder 12,824
         100.7 MHz, 4 cycles, II=1
```

⚠️ **Still "largest TRIED", not "largest that fits".** Nothing failed. The ladder has been
extended to 1600 and 2000 (training in flight) to find the actual edge.

#### The area model was wrong by 2x, and it cost two configs

`1x600` was predicted at **96.9%** and measured at **51.11%**. The cause was the constant 67%
selection ratio, measured at `sm` alone. Across the sweep the true ratio falls monotonically
with width -- 67% at `1x50`, 44% at `1x360`, **25% at `1x1200`** -- because more wiring slots
compete for the same `features x z` thermometer bits.

**That mis-prediction filtered `1x800` and `1x1200` out of the sweep entirely.** Both were re-run
afterwards and both fit comfortably (62.0% and 77.2%).

Replaced with an **occupancy model**: `S` slots drawing from `M` bits give
`M(1-(1-1/M)^S)` distinct if independent, times a concentration factor `c(S/M)` fitted as a
quadratic in `log10(S/M)` because the learnable mapping is not independent.

| | old (constant ratio) | new (occupancy) |
|---|---|---|
| worst error, comparators | **+110%** | **11.1%** |
| mean error, comparators | ~35% | **4.0%** |
| worst error, `dwn_top` area | >100% | 17.7% |

Fitted on 30 configs spanning widths 50-1200, n 2/4/6, z 8-800, 1-3 layers.

#### Results that answer the study's questions

**`z=200` is past the knee -- the paper's unexamined constant, now costed.** At 1x360:

| z | 8 | 25 | 50 | 100 | **200** | 400 | 800 |
|---|---|---|---|---|---|---|---|
| accuracy | 73.11 | 75.27 | 75.61 | 75.75 | **75.85** | 75.81 | 75.77 |
| LUTs | 1,822 | 3,381 | 4,825 | 6,421 | **8,006** | 9,334 | 10,572 |

z=50 gives up **0.24 pp for 40% less silicon**, and z=400/800 are *worse* than z=200 while
costing more. The paper fixes z=200 for every JSC config and never reports its cost.

**The 14x encoder ratio is a small-model artifact, measured across the ladder:**
14.1x → 12.7x → 9.8x → 8.2x → 7.5x → 7.1x → 5.8x → **4.5x** from `1x50` to `1x1200`. And past
~1200 nodes comparators saturate toward the 3200 ceiling, so **the core overtakes the encoder as
the growth term** -- reversing the entire Phase 1 story, exactly where the paper's `lg` sits.

**`n=2` is ON the frontier, contradicting dse-plan §3.** The plan predicted n=2 and n=4 would be
"worse on both axes." At 1x200, n=2 scores 74.06% at **2,319 LUTs** against n=6's 75.32% at
5,036 -- worse accuracy, **2.2x cheaper**, and not dominated. Fewer slots select fewer distinct
thresholds, which shrinks the encoder faster than accuracy falls.

**Single layer wins on accuracy; multi-layer wins on speed.** 1x200 (75.32%) > 2x100 (74.42%) >
3x65 (73.87%), matching the paper. But `2x100` hit **155.5 MHz** and `3x120` 152.3 -- the
fastest in the sweep -- because `PIPE_LUT` inserts a register per layer, so depth buys
pipelining for free.

#### ⚠️ Group B: no reduced-pipeline variant meets the board clock at 1x360

| variant | cycles | Fmax | latency ns | WNS at 10 ns |
|---|---|---|---|---|
| baseline (4-stage) | 4 | 104.6 | 38.24 | +0.440 |
| 3-stage: no OUT reg | 3 | 99.4 | **30.18** | **-0.059** |
| 3-stage: no POP reg | 3 | 72.5 | 41.38 | -3.793 |
| 2-stage | 2 | 66.5 | 30.08 | -5.040 |
| clock 8 ns (125 MHz) | 4 | 124.7 | 32.08 | **-0.020** |

**The lowest-latency variant is the one that does not run on the board** -- missing 100 MHz by
0.6%. And 125 MHz misses by 0.25%. Two results sit agonisingly on the wrong side of a
constraint.

Also: the two "3-stage" variants are *nothing alike*. Dropping the output register costs 5 MHz;
dropping the popcount register costs **32 MHz**. The popcount tree is the critical path, and it
worsens with width -- the same gap was 7 MHz at `1x50` in Phase 1.

⚠️ **Group B ran on ONE rung.** `dse-plan` §6 step 4 says "pipeline/clock sweeps on ~5
already-trained models"; the implementation does five hardware variants of `1x360`. Since
timing tightens sharply with width (147 MHz at `1x50` → 100 MHz at `1x1200`), whether a reduced
pipeline is viable almost certainly depends on size -- and that has been measured at exactly one
size. Cheap to extend: no training, ~5 min per variant.

⚠️ **4 stages is the architectural maximum for a single-layer model** (encoder, LUT layer,
popcount, argmax; `pipe_reg.ENABLE` is 0/1). If a config ever misses timing at 4 stages, the
current RTL cannot rescue it -- but a multi-layer model of similar size could, since each layer
adds a stage.

#### Failures, kept as results

- `1x200 linear`, `1x360 linear` -- **unbuildable at Q3.12**. Evenly-spaced thresholds span the
  data range and reach 8.906, past the +8 ceiling; 23 of 3200 overflow. Q4.11 would represent
  them at *identical* area (still 16-bit), but the encoding axis shows a 0.12 pp spread --
  below the 0.15 pp run-to-run noise -- so it was not worth the precision plumbing. Recorded as
  `gate1-failed` with the reason.

#### Calibration: the noise floor

Phase 1's `1x50` checkpoint and the sweep's own `1x50` are the same config, seed and tau, trained
in different sessions: **73.8361% vs 73.9855%, a 0.15 pp spread.** That is the resolution limit
for every accuracy comparison here -- and it means the encoding differences (0.12 pp) and the
z=400-vs-z=800 "reversal" (0.12 pp) are noise, not signal.

### 2026-08-09 — ⚠️ The encoder narrowing result was fitted and tested on the same data

Phase 1 recorded per-feature comparator narrowing as **-17.1% and "bit-exact against the Q3.12
spec (0 differences across 202 comparators x 1000 samples)"**. The area saving is real. **The
bit-exactness is not established**, and the method is circular:

`analyze_encoder.min_frac_bits()` searches for the fewest fractional bits that reproduce every
comparison **for the samples it is given**, and `experiment_narrow_encoder.py` gives it the
1000-sample test vectors -- then validates the result against those same 1000 samples. Fitting
and testing on one sample of the data.

Re-derived against the full 166k set:

| | frac bits from 1000 | required on 166k |
|---|---|---|
| feature 4 | 6 | **12** |
| features 5, 6, 8 | 8 | **12** |
| feature 9 | 9 | 11 |
| feature 10 | 11 | 12 |
| feature 11 | 10 | 12 |
| feature 12 | 11 | 12 |

**8 of 15 features were narrowed too far.** On the full test set those comparators would produce
different bits than the Q3.12 spec -- so the shipped design would not match the golden model,
and Gate 1b would have found it on hardware rather than here.

**The safe saving is roughly half.** By the linear cost model: -10.7% for the (unsafe)
1000-sample fit against **-8.5%** for widths safe on 166k. Scaling by the model's known
underestimate (it predicted -12.4% where measurement gave -17.1%), a safe narrowing is likely
around **-13%**, not -17.1%.

**Consequences:**

- The "-17.1%, not adopted" line in `docs/phase1-ledger.md` should be read as **-13%, and the
  original was never safe to adopt**. Not adopting it was right for a better reason than the one
  recorded.
- **Narrowing is data-dependent per config.** Widths must be derived from the full test set, not
  a subset, and re-derived for every sweep point -- its thresholds differ. Any future adoption
  has to fit on 166k and re-verify through Gate 1.
- The general lesson is the same one the emitter read-back check taught: **a check that uses the
  same input as the thing it checks proves only self-consistency.**

### 2026-08-09 — The frontier edge can be MEASURED, not predicted

Tested whether Vivado reports utilization for a design that exceeds the part, by synthesizing a
2400-node / 3177-comparator config (the paper's `lg` scale). **It does** -- no error, full report
at **139.28% of the device**:

| module | LUTs | % dev | WNS |
|---|---|---|---|
| `dwn_core` | 5,663 | 27.23% | |
| `thermometer_encoder` | 23,307 | 112.05% | +7.013 |
| `dwn_top` | **28,970** | **139.28%** | **-0.948 (fails 100 MHz, 91.3 MHz)** |

Only *place-and-route* fails on an over-budget design; synthesis alone gives measured area. So
`--measure-filtered` now synthesizes (never implements) a too-big config, turning
*"predicted 128% of device"* into *"measured N LUTs at 76.20% accuracy, does not fit"* -- which
is the claim brief §12 risk #2 asks for. Off by default; a normal sweep should not spend Vivado
time on configs it has already rejected.

`measure_only()` is deliberately a separate function rather than a flag on `run_config`, because
it skips **Gate 1**. That is safe only because such a config enters the frontier as *the point
where the part runs out*, not as a working design -- area is the claim, correctness is not.
`run_config` must keep gating synthesis on Gate 1, and mixing the two paths would erode that.
Verified against `1x50`: returns exactly 108 / 1519 / 1619.

#### The area model extrapolates well -- 7x past its calibration range

The same run is an independent check on `dse/area_model.py`, which was fitted on 50-360 nodes:

| | predicted | measured | error |
|---|---|---|---|
| encoder | 24,063 | 23,307 | **+3.2%** |
| dwn_top | 30,624 | 28,970 | **+5.7%** |
| core | 6,561 | 5,663 | +15.8% |

The core term now *over*-estimates at extreme width (the `0.13·log2(group/10)` slope, fitted on
groups of 10-72, extrapolates hard at group=480). **That is the safe direction for a filter** --
it errs toward skipping a config that might have fit, which the probe band then catches.

⚠️ **Not added as a calibration point.** The design had random tables, and random tables are
maximally incompressible, so 5,663 is an upper bound on what a trained model of that size would
cost. Useful as validation, wrong as a fitting point.

### 2026-08-09 — Filtered configs keep their accuracy

`run.py` bailed out of a too-big config before reading its checkpoint, so `1x800` and `1x1200`
would have produced rows of dashes despite being trained. `1x800` reached **76.20%**.

Now a filtered config still records accuracy when a checkpoint exists. The point is not
tidiness: **"76.20% is achievable but needs 128% of the device" is the frontier-edge datapoint
Study 1 owes** (brief §12 risk #2). Without it the accuracy-vs-width curve stops at `1x600` and
says nothing about the wall it is supposed to locate -- while the measurement sat unread in a
file we had already spent GPU time producing.

Costs nothing: it only reads a checkpoint for configs that were being skipped anyway, and it
degrades to "no checkpoint, so no accuracy -- area prediction only" when the file is absent.

### 2026-08-08 — The deliverable figures were broken at full grid size

`report.py` and `plot.py` are what Study 1 actually hands over, and they had only ever run on
1-3 configs. Exercised on a synthetic 37-config results file (`--results`, added for this and
useful on its own for inspecting an alternate run). Four defects, none of which would have
appeared before the sweep finished:

**1. Two report branches had never rendered.** The 3-objective frontier and the "constrained
clock NOT met" list both need data shapes that did not exist yet. Both work: the latency view
adds 6 points over the 2-objective one, correctly separating Group B (`1x360 2-stage` at
**17.20 ns** against 34.39 for the 4-stage), and the timing list catches a config at WNS -0.400.

**2. The frontier legend covered a frontier point.** Placed lower-left, which is exactly the
corner a minimize-area/maximize-accuracy frontier reaches into. At 3 configs it was empty; at 35
it hid the cheapest point on the frontier. Legend moved OUTSIDE the axes.

**3. The device-ceiling label collided with the largest config's label.** Top-right is where the
biggest frontier point and its direct label land. Ceiling label moved to the bottom of the line,
plus headroom so a top label is not clipped.

**4. `area_split` was unreadable — and it was a FORM error, not a layout one.** It plotted all 35
configs, so 35 rotated labels overlapped and `12.6x` printed five times on top of itself (the
Group B configs have *identical* area to their base rung, by definition). The figure's job, per
its own docstring, is the core/encoder split **across the size ladder**; the one-factor variants
sit at two fixed widths and Group B adds nothing. Restricted to the ladder: six bars, readable,
and the trend is legible (14.1x -> 12.2x as width grows).

#### A silent mis-classification found while fixing #4

`group_of()` guessed the grid group from the label — *"one-factor if the label contains a
space"*. Every multi-layer config (`2x100`, `3x65`, `2x180`, `3x120`) has no space in its label,
so all four were classified as **ladder** points: mis-coloured on the scatter and pulled into
the ladder-only bar chart. Fixed by recording the actual grid group in the result and reading
it, with the heuristic kept only as a fallback for older records.

### 2026-08-08 — Gate 1 at n=4 and n=2, without training either

`scripts/make_test_checkpoint.py`. The n=2 packing fix above was verified only in Python, and
CLAUDE.md is explicit that this does not count -- "confidence is not verification", with no
exceptions for code Claude wrote. But no n!=6 checkpoint exists and training is on Kaggle.

**Gate 1 does not need a TRAINED model.** It asks whether emitted RTL matches the golden
software model, and both derive from the same checkpoint. Random tables exercise that machinery
as well as learned ones -- better, since they hit address patterns a trained model might never
produce. So: fabricate a checkpoint at any `n`, compute reference predictions with the golden
model through the real quantize -> encode path, write the `_testvectors.npz`, run the normal
flow. (What this cannot check is numpy-vs-PyTorch agreement -- but `n` dependence does not live
there.)

```
n=4   core 1504/1504   top 1519/1519   PASS
n=2   core 1504/1504   top 1516/1516   PASS
```

**First time n!=6 RTL has ever been simulated.** Four grid configs depend on it.

#### The test was then verified to be capable of failing

A passing test proves nothing until it can fail. Re-running n=2 with the packing bug restored:

| packing | mismatches | verdict |
|---|---|---|
| old (`np.packbits`) | **958 / 1504** | FAIL |
| fixed | 0 / 1504 | PASS |

#### ⚠️ And that run exposed a limit of the emitter's self-check

With the bug present, `emit_core.py`'s read-back check still reported **20/20 nodes match**.

It parses its own emitted Verilog and compares the tables against `table_to_hex(...)` -- **the
same function that had the bug**. The check is circular: it proves the file says what the emitter
meant, never that the emitter meant the right thing. Only Gate 1, with an independent golden
model, catches this class of error. `emit_core.py` already says as much in its closing message
("proves the file SAYS what the checkpoint says... proves nothing about how the Verilog
BEHAVES") -- that is now demonstrated rather than asserted, and it is the reason Gate 1 is
non-negotiable rather than belt-and-braces.

### 2026-08-08 — Multi-layer verified, not assumed

I had asserted Phase 1 proved multi-layer emission works. **It did not** — Phase 1 ran
`extract.py` over the `[300,100]` model but never emitted or simulated RTL from it. Four grid
configs are multi-layer (`2x100`, `3x65`, `2x180`, `3x120`), so that was an untested path
carrying real sweep points.

Tested with a Phase 1 checkpoint that was already on disk (`t8_distributive_300-100`, n=6, z=8):

```
400 lut_node instances, 128-bit input, 2 layer(s)
core latency 4 cycles (= PIPE_LUT x 2 + POP + OUT), top latency 5
read-back: 400/400 nodes, 124/124 comparators
GATE 1: core 1504/1504, top 1519/1519, PASS
```

Three things this covers that nothing else did:

- **`emit_core` looping over layers** — layer 0 into layer 1 into the popcounts.
- **Latency arithmetic at L>1.** Core latency is 4, not 3, because `PIPE_LUT` applies per layer.
  Had the golden model and the RTL disagreed here, every vector would have been compared against
  the wrong cycle — the exact bug class that has bitten this project twice.
- **Both wiring representations.** This checkpoint declares one mapping for two layers, so layer
  0 resolves through `.mapping.weights` (checkpoint-format §3a, argmax) while layer 1 falls to
  the fixed-wiring path (§3b), which had never been through Gate 1. `extract.py` warns that
  `__dummy_mapping` "has the same shape and dtype as a real mapping", so keying off shape yields
  "a valid-looking, totally wrong export" — silent, like the n=2 packing bug.

### 2026-08-08 — ⚠️ LUT tables were emitted shifted at n=2

Found by auditing what the grid uses that has never executed: **every config emitted or simulated
so far is n=6**, but the grid has four at n=4 and n=2.

`table_to_hex()` built the table with `np.packbits`, which pads a partial byte on the **low**
side under `bitorder='big'`. A table shorter than 8 entries therefore came out shifted left. At
n=2 the 4-entry table `[1,1,1,0]` emitted as `0x70` instead of `0x07` — **every entry at the
wrong address**. n≥3 was unaffected, because `2**n` is then a whole number of bytes, which is
exactly why n=6 never showed it.

**This was more dangerous than an ordinary off-by-one.** `docs/dse-plan.md` §3 *predicts* n=2
fails — routing congestion — and states that such a failure "is a data point marking the
frontier's edge, not a mistake to prevent." A Gate 1 failure caused by this bug would have
looked exactly like the expected architectural finding, and could plausibly have been written up
as one. Gate 1 would have caught the mismatch; nothing would have caught the misattribution.

Rewritten as an explicit loop over addresses. n ≤ 6, so at most 64 iterations per node —
clarity beats vectorization, and the previous version's cleverness is what hid the bug.

**Verified both directions:**
- correct at n = 1, 2, 3, 4, 5, 6 (bit `addr` == `row[addr]`)
- **byte-identical to the old packing for every n ≥ 3** across 200 random tables each, so Phase
  1's RTL and every recorded number are unchanged — confirmed by re-running **Gate 1: 1504/1504
  core, 1518/1518 top, PASS**

### 2026-08-08 — ⚠️ `tau` interpolation was wrong; training restarted

Caught before the sweep produced anything, but only just — the first full Kaggle run was already
in flight and had to be abandoned.

**The paper's tau is a power law in width, not a linear function of it.** Its four JSC anchors
have log-log slopes of 0.526, 0.557 and 0.635 — near-constant, i.e. `tau ≈ width^0.57`.
`tau_for()` interpolated linearly *in tau* between anchors, which overshoots everywhere between:

| nodes | was | correct | error |
|---|---|---|---|
| 100 | 5.674 | 4.902 | +15.7% |
| 200 | 8.015 | 7.210 | +11.2% |
| 500 | 14.040 | 12.318 | +14.0% |
| 600 | 16.283 | 13.829 | +17.7% |
| 800 | 19.821 | 16.599 | **+19.4%** |
| 1200 | 24.808 | 21.470 | +15.5% |

**Why this was worse than a uniform offset.** 50 and 360 are exact anchors, so they got the
paper's value while every other rung ran 10–19% hot. The error is a *wiggle* that returns to
zero at the anchors — so the accuracy-vs-width curve would have carried a kink that is an
artifact of the interpolation, in the one place it does the most damage: the size ladder is the
spine of the frontier and the source of the headline number.

This is exactly the failure this ledger already warned about — *"getting this wrong does not
fail loudly; it just trains a worse model, and the sweep point then reports an accuracy that
says more about tau than about the architecture."* Fixed by interpolating **log(tau) linearly in
log(width)**; both anchors reproduce exactly (50 → 3.3333, 360 → 10.0).

**Training restarted from zero**, in a fresh Kaggle notebook. Of what had been trained, only
`1x360 z=8` was unaffected (360 is an anchor). Keeping it was not worth the risk: the notebook
resumes on file existence, so a single stale checkpoint would silently survive with old tau and
the resulting frontier would mix two schedules with nothing in the output revealing it. A fresh
notebook also matters because the old one's saved versions have the wrong grid embedded — an
old version re-run later would quietly reproduce the bug.

**Second defect, found while restarting: resume did not survive a session.** The check was
`os.path.exists('/kaggle/working/<slug>_checkpoint.pt')`, but `/kaggle/working` is fresh on every
Kaggle *version* — it persists only within a live session. So a 32-config run that cannot fit
one session would have restarted from zero each time, silently doing the work again. The
notebook now copies any `*_checkpoint.pt` / `*_testvectors.npz` found under `/kaggle/input`
forward into `/kaggle/working` at startup, so adding the previous run's output as an input
dataset makes resume work across sessions and leaves the final Output panel holding the complete
set rather than one session's slice.

### 2026-08-08 — Group B paths validated before the sweep needs them

Every run to that point had `pipe=1111, clock_ns=10.0`. All five Group B configs ride on
`1x360`, which is not trained yet — so two paths had **never executed**: `cfg.hw.pipe_*` through
`run_config` → `gate1()` → the emitters, and `cfg.hw.clock_ns` through `run_one(period=)` →
`build.tcl`. Exercised deliberately against the baseline checkpoint, and both reproduce numbers
Phase 1 already measured:

| | measured | Phase 1 |
|---|---|---|
| 3-stage (`pipe_out=0`) latency | **3** cycles | 3 |
| 3-stage Fmax | **122.9 MHz** | 122.9 MHz |
| 3-stage WNS | **+1.864** | +1.864 |
| 4-stage WNS at **8 ns** | **+1.790** | +3.790 at 10 ns |

Gate 1 passed at latency 3 for the pipeline variant, so a swept depth is *verified*, not merely
built.

⚠️ **The clock check only works on WNS, and this nearly hid itself.** Fmax was **161.0 MHz at
both 8 ns and 10 ns** — correctly, since it reflects the same critical path. A dropped `period`
argument and a working one are **indistinguishable on Fmax**. Only WNS separates them, and
+3.790 → +1.790 is exactly the 2.000 ns of budget removed. Had the argument silently not reached
Vivado, both clock configs would have reported the baseline's numbers and the clock axis would
have read as *"target clock has no effect"* — a plausible-looking false finding.

The two ad-hoc rows were removed from `build/dse/results.json` afterwards: that file should
mirror `dse/grid.py`, or the record becomes ambiguous about what was actually swept. Group B on
the baseline is a legitimate sweep point if wanted — but then it belongs in the grid, not in the
results as a leftover.

### 2026-08-08 — 2d: the area filter, with a probe band that keeps the wall measurable

**2d was never actually implemented.** `dse/grid.py` reported "will synthesize: 34" as a budget
projection, but `dse/run.py` ignored it and would have synthesized every config with a
checkpoint — including `1x1200` at a predicted 133% of the device, which is a long
place-and-route to reach a foregone conclusion.

Now `grid.should_synthesize(cfg) -> (run?, reason)`, used by both the budget summary and the
runner, so the projection and the behaviour cannot disagree.

**The filter deliberately does not skip everything that overshoots**, because that would defeat
Study 1. If every predicted-overshoot config is skipped, **nothing ever fails to fit** — the
headline stays "the largest we *tried*" forever, and the frontier's edge ends up predicted
rather than measured. Brief §12 risk #2 is explicit that a config that cannot fit or route is a
data point the study is meant to produce.

So a config is synthesized when it is predicted to fit, **or** its estimate is extrapolated
(n≠6 or z≠200 — an overshoot there is not evidence), **or** it lands inside a probe band up to
**115% of the device**. That ceiling is tied to the model's own error, not chosen for
roundness: 0.5% at the calibrated point but ~9.5% where it extrapolates, so anything within
~10–15% of the threshold could genuinely go either way. Past that the prediction is outside its
own error bars and skipping is safe.

Current effect on the ladder:

| Config | predicted | decision |
|---|---|---|
| `1x600` | 96.9% | **RUN** — probe, inside the error band |
| `1x800` | 127.6% | skip |
| `1x1200` | 132.9% | skip |

So `1x600` is the config that settles the wall: either it fits and the model is pessimistic, or
it fails and the edge is **measured**. Either is a result; skipping it produces neither.

Filtered configs are **recorded as `filtered-too-big` with their reason**, appear in the report's
table and its "not synthesized" section, and are never silently dropped. `--no-filter` forces
them through.

### 2026-08-07 — Sweep results are committed evidence; the weights are not

`dse/report.py --snapshot` writes `docs/results/sweep-results.{json,csv}`.

**The gap this closes.** `build/dse/results.json` is gitignored with the rest of `build/`, on the
rule that everything there is regenerable. But it is not regenerable in any useful sense —
reproducing it costs a Kaggle GPU session plus ~7 h of Vivado. It is the **record of what was
measured**, and without it the repo can show what the sweep *planned* (`dse/grid.py`) but not
what it *found*.

**And it is the right thing to commit, rather than the checkpoints.** A few hundred KB of JSON
carries every claim: config identity (n, z, encoding, widths, pipeline, clock), accuracy, core
and encoder LUTs **separately**, device %, Fmax, latency, and whether the row is post-route or
post-synthesis. The 933 MB of weights it describes is not needed to support any of that. ~10 KB
at 37 configs.

### 2026-08-07 — First sweep points, and the area model recalibrated on them

The two z=8 configs from the first Kaggle session, measured end to end. **The first sweep data
that is not the Phase 1 baseline**, and it immediately corrected the area model twice.

| Config | acc | core | encoder | top | % dev | Fmax |
|---|---|---|---|---|---|---|
| `1x50` (z=200) | 73.84% | 108 | 1519 | 1619 | 7.78% | 147.1 MHz |
| `1x200 z=8` | 72.68% | 466 | 879 | 1345 | 6.47% | 118.8 MHz |
| `1x360 z=8` | 73.18% | 865 | 970 | 1835 | 8.82% | 120.0 MHz |

**The frontier had to discriminate for the first time, and did.** `1x360 z=8` costs *more* area
than `1x50` for *less* accuracy, so it is dominated and correctly dropped; `1x200 z=8` is
cheaper and less accurate, so it stays. Two points on the frontier, one off it.

**Early signal on the z hypothesis, with a caveat.** Dropping z from 200 to 8 — a 25× cut in
input bits — costs only **~0.7 pp** (73.84% → 73.18%, though at 360 nodes rather than 50).
That is the direction the sweep was built to test. It is not yet a result: the widths differ, so
the clean comparison is `1x200 z=8` against `1x200 z=200`, which needs the z=200 ladder.

#### Correction 1: the core term was underestimating by up to 19%

A flat **1.0 LUT per final-layer bit**, fitted on `sm` alone, does not survive wider layers:

| Config | group | reduction | LUT per final bit |
|---|---|---|---|
| `1x50` | 10 | 58 | 1.00 |
| `1x200 z=8` | 40 | 266 | 1.27 |
| `1x360 z=8` | 72 | 505 | 1.36 |

Cost per bit **rises with group width**, and it has to — a popcount is an adder tree, and wider
groups mean more levels carrying wider adders. A constant was always going to be wrong away from
the width it was fitted at; one data point simply could not show it. Now
`1.0 + 0.13·log2(group/10)`, with the argmax term scaling on score width as `(K-1)·W/2`.

**Core error: +14% / +19% → under 0.5% on all three configs.**

This mattered: `1x600` sits at 96.9% of the device against a 90% fit threshold, so a 20% core
underestimate is the difference between synthesizing a config and skipping it. No fit decision
actually flipped — the wall stays between `1x500` (81.1%) and `1x600` — but the margins are now
honest.

#### Correction 2: encoder saturation is `used_features × z`, not `features × z`

`1x200 z=8` predicted 963 encoder LUTs and measured **879** — about 117 comparators where the
model assumed all 128 (16 features × 8 bits). `1x360 z=8` *did* reach ~128.

So the ceiling is not the number of thermometer bits that exist, it is the number belonging to
features the mapping actually **reads**. Phase 1 already recorded `d2_b2_mmdt` as never read at
all; about 1.4 unused features at 200 nodes accounts for the 11 missing comparators exactly.

**Not fixed, deliberately** — which features get used is learned, so it cannot be predicted
without training the config. This is the same limit `is_extrapolated()` already marks, and it is
why `dse/grid.py` never filters on an extrapolated estimate.

#### The self-test now separates the two kinds of wrong

It validates all three measured configs, and **fails only on calibrated-point error** (n=6,
z=200), reporting extrapolated error separately. Holding an extrapolation to the same bar would
force either a dishonest fit or a tolerance loose enough to hide a real regression where the
model claims to be accurate. Currently: **0.5% calibrated, 9.5% extrapolated.**

### 2026-08-07 — `--impl` validated through the runner

The last untested path in the sweep pipeline. Everything so far had gone through
synthesis-only, but `--impl` is what every real sweep point will use.

```
core 108 | encoder 1519 | top 1619 | Fmax 147.1 MHz | 379s
```

**147.1 MHz is Phase 1's post-route figure exactly** (against 161.0 post-synthesis), so the flag
genuinely reaches Vivado through `run_one()` rather than being silently dropped. Area identical.
The checkpoint resolved **by slug with no `--checkpoint` argument**, exercising the resolution
path added in 6d. `report.py` correctly dropped its post-route caveat once `impl` was true, and
latency was restated as **27.19 ns** (4 cycles at 147.1 MHz) instead of the optimistic 24.84 at
post-synthesis Fmax — which is the point of computing latency in nanoseconds.

**Cost: 379 s = 6.3 min per config**, against the 12 min budgeted. `MINUTES_PER_SYNTH` is
deliberately **not** lowered: 6.3 min is the *smallest* config in the grid, `1x1200` has 24× the
nodes, and place-and-route scales worse than linearly with occupancy. Planning the ladder from
its cheapest rung would underestimate exactly the end that runs a session out of time.

### 2026-08-07 — The reduction, measured: 50 + 58 = 108 exactly

`scripts/experiment_reduction.py` synthesizes the two halves of `dwn_core` separately, closing
an open question Phase 1 left and `dse/area_model.py` depended on.

| Part | LUTs | % of core |
|---|---|---|
| `nodes_only` — 50 `lut_node`, real tables and wiring | **50** | 46.3% |
| `reduction_only` — 5 × popcount(10) + argmax | **58** | 53.7% |
| **sum** | **108** | **100.0%** |
| `dwn_core` measured in Phase 1 | 108 | |

**Two results, and the first is the more valuable.**

**1. One DWN node is exactly one LUT6 — measured, not assumed.** 50 nodes, 50 LUTs. Brief §4's
architectural premise, and the thing the whole area model rests on, had never been observed
directly because Vivado inlines `lut_node` into the top level. Now it has.

**2. The inferred reduction was exactly right, and the sum is exact.** 108 = 50 + 58 with *zero*
cross-boundary optimization — the fragments do not share logic with each other in the real core.
So `area_model.py`'s reduction term (1.0 LUT per final-layer bit + 1.6 per class) is
measurement-backed rather than a fitted guess.

⚠️ **But do NOT read "54% of the core" as "worth building Learnable Reduction".** The script
prints that verdict because `dse-plan` §3 set the bar at *"40% of area"* — and that bar was
written before anyone knew the encoder costs 14× the core. Judged against the whole design:

| | `sm` 1×50 | `md` 1×360 (projected) |
|---|---|---|
| reduction | 58 | ~368 |
| dwn_top | 1,619 | ~12,101 |
| **reduction as % of design** | **3.6%** | **~3%** |

**Eliminating the reduction entirely would save ~3.6%** of a design the encoder dominates.
dse-plan's own framing — *"if it's 3%, this was never an interesting axis"* — is the one that
applies once the denominator is the real design rather than the core. The 40% figure is not
wrong, it is measuring the wrong ratio.

**Recommendation: leave Learnable Reduction deferred.** It is unimplemented upstream, so it is
custom training code plus new RTL plus a Gate 1 re-run, for ~3% of area on the axis that is not
binding. `z` and per-feature narrowing both move far more. Revisit only if a config lands
marginal and the encoder levers are exhausted.

*(Caveat carried from the script: fragments synthesized alone cannot share logic with
neighbours, so both figures are upper bounds on in-context cost. That the sum lands exactly on
108 says there was nothing to share here.)*

### 2026-08-07 — 2c: the grid training notebook

`training/train_grid_kaggle.ipynb`, plus `dse/grid.py --json` so the notebook consumes the grid
rather than restating it. `dwn_jsc_kaggle.ipynb` is untouched — it reproduces the paper's `sm`
and is the Gate 1 reference; it should not grow a loop.

**Binarization is grouped, and the saving is large.** 32 configs need only **9** distinct
`(encoding, z)` binarizations, because 16 of them — the entire width ladder, the `n` sweep and
the layer-count sweep — share `(distributive, 200)` and differ only in model shape. Binarizing
per config would repeat the most expensive setup step 16 times for byte-identical output.

**z=800 does not fit, so there are two data paths.** Measured sizes for the binarized set
(830k samples × 16 features × z, uint8):

| z | 8 | 25 | 50 | 100 | 200 | 400 | **800** |
|---|---|---|---|---|---|---|---|
| GB | 0.11 | 0.33 | 0.66 | 1.33 | 2.66 | 5.31 | **10.62** |

Above 6 GB the notebook binarizes **on the GPU per batch** instead of precomputing. The risk is
obvious — two code paths producing different bits would make a large-z config incomparable to a
small-z one, and the difference would look like a *result*. So the precompute path **asserts the
GPU path reproduces it exactly** on 256 samples before training anything. It is a
speed/memory tradeoff, never a correctness one.

**Resumability is mandatory, not a nicety.** 32 runs will not fit one Kaggle session (9 h cap,
30 h/week quota). The notebook skips any config whose checkpoint already exists, orders configs
cheapest-first within a group so a session that dies late still banks the most models, and has
`ONLY_N` to bound a session deliberately rather than by timeout.

**Schema verified against Phase 1 rather than assumed.** The checkpoint `config` dict carries
exactly the same 13 keys as the Phase 1 checkpoint — none missing, none extra — and includes all
four the exporter actually reads (`layers`, `n`, `num_classes`, `thermometer_bits`). The
`_testvectors.npz` carries `x_binarized` / `x_raw` / `pred`, which is what `gen_vectors.py`
consumes, under the `<slug>_testvectors.npz` name it derives from the checkpoint path.

**Filenames are the contract**: `<slug>_checkpoint.pt`, from `ModelConfig.slug`. `dse/run.py`
resolves by exactly that, so renaming the downloads makes the sweep silently find nothing.

**Deliberately NOT produced: a 166k test-set dump per config.** Gate 1b is Phase 1's exit
condition and was done once for the reference config; Phase 2 takes area and timing from Vivado
reports and accuracy from the checkpoint. Per-config dumps would be ~300 MB that nothing reads.
If a specific sweep config is ever taken to hardware, `dump_testset_kaggle.ipynb` handles it.

⚠️ **Not yet run.** Every cell parses and the schema is verified, but nothing here has executed —
`torch_dwn` has no CPU path, so the notebook cannot be tested off a GPU. Expect the first Kaggle
run to be the real test, and expect training time to be the open question: 32 configs × 32 epochs,
with the larger rungs slower than `sm`.

### 2026-08-07 — 6d: results, frontier, plots — and three bugs it caught first

`dse/report.py` + `dse/plot.py`. Building the reporting layer **before** 2c was the right order:
it found three defects that would each have been baked into 32 training runs' worth of results.

**1. No accuracy in the result schema.** The record had area and timing but nothing to put on
the other axis — a Pareto frontier over (accuracy, area) was literally impossible. Now read from
the checkpoint, so it cannot be attached to the wrong config.

**2. Accuracy units were wrong by 100×.** The checkpoint stores `final_acc` as a *fraction*
(`0.7383614`), while every ledger, table and paper comparison here quotes *percent*. The first
report printed **0.74%**. Converted once at the source, fields renamed `accuracy_pct` /
`accuracy_best_epoch_pct` so no consumer has to guess, and an **assertion** rejects anything
outside [0,1] — if a checkpoint ever stores percent it must fail loudly, not report 7384%.
(`final_acc` is the primary number: the saved weights are the final epoch and `best_acc`
describes weights that were never saved.)

**3. ⚠️ A silent-corruption footgun in `dse/run.py`.** `run_config()` emits RTL from the
checkpoint and uses `cfg` only for naming and area prediction. So `--all --checkpoint <one>`
would have emitted **one model's RTL under all 37 config names** — every row wrong, nothing
failing, and the frontier would have looked plausible. Fixed three ways:

- `ModelConfig.slug` (`n6_z200_distributive_w50`) is the identity of the *trained model*:
  n, z, encoding, widths, and deliberately **no hardware params**, so all five Group B configs
  share one checkpoint instead of demanding five identical training runs.
- `resolve_checkpoint()` looks up `<slug>_checkpoint.pt` per config; no match means skip with
  the expected filename printed, never a wrong build.
- `--checkpoint` with more than one config is now a **hard error**.

This also fixes the naming contract for 2c: the notebook must write `<slug>_checkpoint.pt`.
Phase 1's checkpoint predates the convention and is mapped by an alias rather than renaming a
file every recorded number refers to.

**Latency is a real axis now.** `pareto()` is generic over objectives, and latency is computed
in **nanoseconds** (`cycles / Fmax`), not cycles. Cycles alone cannot rank pipeline variants:
Phase 1 measured 4 stages at 161.0 MHz = **24.84 ns** against 3 stages at 122.9 MHz = 24.4 ns —
a whole cycle apart and effectively the same real latency. Ranking on cycles would call that a
clear win; ranking on time shows it is a wash. Cycles are still reported (brief §6). Under the
2-objective view all five Group B configs tie; the 3-objective view is what separates them, so
the report prints both and says which points only the second view reveals.

**Plots** (`matplotlib==3.10.7` added to requirements — nothing else imports it, so a missing
matplotlib degrades plotting only). Two figures into `build/dse/`: `frontier.png` (accuracy vs
area, Pareto line, device ceiling) and `area_split.png` (core vs encoder stacked, per rung).

Two constraints worth recording because they are not aesthetic:

- **Scatter caps at three categorical colors** (all-pairs CVD separation). The grid's six groups
  fold to ladder / one-factor / group-B — which is also the distinction that carries meaning.
- **The device ceiling must not set the y-scale on `area_split`.** Anchoring the axis to 20,800
  squashed a 1,619-LUT design into a sliver and made the 108-LUT core *invisible* — destroying
  the one thing the figure exists to show. It now scales to the data and states the ceiling in
  words when it is off-scale. Caught only by rendering the PNG and looking at it; the palette
  validator checks color, never layout.

### 2026-08-07 — 2a step 6c: the sweep runner, and 2a is done

`dse/run.py` plus a refactor of `scripts/run_gate1.py` that extracts Gate 1 from `main()` into an
importable `gate1(ckpt, vivado_bin, rtl_dir, work, pipe) -> (ok, info)`. `main()` is now a thin
CLI wrapper over it. `dse/` imports it rather than shelling out and scraping stdout — the reason
`run_xsim()` was exposed back in Phase 1.

**The baseline config went through the whole pipeline and reproduced Phase 1 exactly:**

| Module | LUTs | FF | WNS | Fmax | Phase 1 |
|---|---|---|---|---|---|
| `dwn_core` | 108 | 73 | +3.790 | 161.0 MHz | identical |
| `thermometer_encoder` | 1519 | 0 | +7.013 | 334.8 MHz | identical |
| `dwn_top` | 1619 | 269 | +3.790 | 161.0 MHz | identical |

This is the acceptance test for 2a as a whole. The numbers were reached by a **different path**
than Phase 1 took — RTL emitted into `build/configs/<name>/rtl`, include dirs derived from the
sources by the rewritten `build.tcl`, clock from `cfg.hw.clock_ns`, Gate 1 called as a function
— and landed on the same values. The config-driven flow is equivalent to the flow every Phase 1
number came from, which is what makes sweep points comparable to them.

**Three rules are enforced in code, not left to discipline**, because breaking any of them
corrupts the frontier quietly rather than failing:

1. **Gate 1 gates synthesis.** A config failing Gate 1 is recorded `gate1-failed` and is *not*
   synthesized. Area for unverified RTL describes nothing (dse-plan §1), and the row stays
   visible rather than becoming a config nobody notices is missing.
2. **Failure to build is a RESULT.** Recorded `synth-failed` and kept — that is where the
   congestion wall is (brief risk #2), and it is a thing Study 1 measures.
3. **Core / encoder / top recorded separately, always** (brief §6). A total-only table would
   have hidden the 14× encoder finding entirely.

**Resumability is deliberate.** 34 points is several sittings on one machine (CLAUDE.md), so
results are keyed by config name in `build/dse/results.json`, done configs skip unless
`--force`, and writes go through a temp file — an interruption at point 20 must not cost points
1–19.

**Measured cost: 288 s per config** for three targets, synthesis only. `MINUTES_PER_SYNTH = 12`
in `grid.py` is the *implementation* figure (Phase 1's `--impl` runs), so the 6.8 h budget
estimate holds for post-route and the synthesis-only pass is more like 2.7 h. The sweep should
use `--impl` for anything quotable: post-synthesis timing uses estimated routing and is
systematically optimistic, and Phase 1 already found post-route to be 147.1 vs 161.0 MHz.

### 2026-08-07 — 2b: the area model, recalibrated and honest about what it cannot predict

`dse/area_model.py`. dse-plan §5 assumed the encoder costs "up to 3.2× the core"; measured is
**14.06×**. Filtering 2d with the old number would have underestimated encoder area by **4.4×**
at `sm` alone.

Reproduces every Phase 1 measurement — comparators, core, encoder exact; `dwn_top` and board
+0.5%, which is real rather than rounding (measured 1619 < core+encoder 1627, so Vivado
optimizes slightly across the module boundary).

**Projected size ladder**, and it revises an earlier ledger entry:

| Config | Nodes | Comparators | Core | Encoder | Board | % dev | Fits |
|---|---|---|---|---|---|---|---|
| `sm` 1×50 | 50 | 202 | 108 | 1,519 | 2,066 | 9.9% | ✅ |
| `md` 1×360 | 360 | 1,454 | 728 | 10,934 | 12,101 | **58.2%** | ✅ |
| `lg` 1×2400 | 2400 | 3,200 (sat.) | 4,808 | 24,063 | 29,310 | 140.9% | ❌ |

⚠️ **Correction: `md` was previously recorded as "≲16,000 encoder LUTs, ~80% — tight".** That
used the *upper bound* of 2,160 comparators (every wiring slot distinct). Applying the measured
67% selection ratio gives 1,454 comparators and **58.2%** — `md` is a comfortable rung, not a
marginal one, which makes it the natural centre of the ladder rather than its edge.

`lg` confirms as not fitting, and starkly: its **encoder alone (24,063) exceeds the whole part**,
while its core (4,808) would fit easily. Note also that `lg`'s encoder/core ratio is **5.0×**,
not 14× — further confirmation that 14× is an artifact of `sm`'s unusually tiny core, not a
constant to extrapolate.

**What this model CANNOT do, discovered while building the grid.** The 67% selection ratio is
**not a collision statistic**. If the mapping picked slots uniformly from the 3,200 available
bits, occupancy would predict ~286 distinct comparators, not 202. The 84-comparator gap is
**learned concentration** — four features carry 153 of 202. How hard the mapping concentrates
depends on how many thresholds each feature has (**z**) and how many slots each node has (**n**),
so a ratio measured at one `(n=6, z=200)` point cannot be transported to other values of either.

Consequence, and it is load-bearing for 2d: **`z` is simultaneously the axis the sweep most wants
to characterize and the one this model is least able to predict.** `is_extrapolated(n, z)` marks
those estimates, and `dse/grid.py` **refuses to skip a config on an extrapolated overshoot** — it
skips only when the prediction is at the calibrated point. Otherwise the filter would silently
discard exactly the configs Study 1 exists to measure.

The softer half is the **reduction term** (~58 LUTs, inferred by subtracting 50 nodes from the
108-LUT core). Vivado inlined `popcount`/`argmax`, so that split is arithmetic, not observation.
The standalone-synthesis open question stands.

### 2026-08-07 — 2a step 6b: the sweep grid

`dse/grid.py` — 37 configs: 8 ladder rungs, 24 one-factor points on two mid-ladder rungs
(z, encoding, n, layer count), 5 Group B variants. Group B rides on the baseline rung's trained
model — same `ModelConfig`, different `HardwareConfig`, no retraining. That is the payoff for
splitting the two objects in step 1.

**`z` gets six values (8, 25, 50, 100, 400, 800), spanning both regimes on purpose.** They
answer different questions, and the area model predicts neither:

| | z=8 | z=25 | z=50 | z=100 | z=400 | z=800 |
|---|---|---|---|---|---|---|
| `1×200` | 8.7% | 18.5% | 33.0% | 33.3% | 33.3% | 33.3% |
| `1×360` | 10.2% | 20.1% | 34.5% | 58.2% | 58.2% | 58.2% |

Below the binding point (`16z < slots × ratio`) encoder area is set by `z` directly. Above it
the model says area is **flat**, because comparators become slot-limited. **Either outcome is a
result:** if flat holds, accuracy above z≈50 is free, which would be the sweep's headline; if it
does not, the 67% selection ratio is climbing toward 1.0 as collisions get rarer, and the area
model needs a z-dependent ratio. The two rungs cross the transition at different z, which is
what makes the flat region testable rather than assumed.

- **The ladder brackets the wall rather than stopping short of it.** `1×500` fits at 80.0%,
  `1×600` fails at 95.6%, `1×800`/`1×1200` fail with the encoder saturated at 3,200. A config
  that does not fit is a data point marking the frontier's edge (brief risk #2), so it is
  reported, never hidden.
- **`tau` interpolates the paper's schedule in log-width**, not copied from `sm`. Getting it
  wrong fails silently — it just trains a worse model, and the point then reports an accuracy
  that says more about tau than about the architecture.
- **Budget: 34 synthesis points ≈ 6.8 h**, against dse-plan §6's 40–70 runs / 15–25 h on one
  machine, plus 32 training runs (Kaggle, step 2c). Still ~2× headroom. Remaining places to
  spend it, in rough order of value: **finer ladder spacing near the wall** (500→600 is a single
  100-node jump and the fit boundary is inside it), a **third OFAT rung**, and `n`×`z`
  interaction points — since those are the two axes the area model cannot extrapolate over,
  and it currently assumes they are independent.

### 2026-08-07 — 2a step 5: the restructure

The move the handoff specified, now done. `emit_core.py` and `emit_encoder.py` are in `rtlgen/`
(via `git mv`, so history follows); `extract.py` stays in `exporter/` per brief §11's split, as
do the `analyze_*`/`experiment_*` scripts. **`rtl/gen/` is deleted, not gitignored** — output
goes to `build/rtl` (default) or `build/configs/<name>/rtl` (swept), both under the
already-ignored `build/`. Exactly as the handoff put it: "it doesn't get gitignored, it stops
existing."

**`scripts/build.tcl` was the only non-mechanical part.** It hardcoded
`-include_dirs rtl/gen`. Include paths are now **derived from the source files actually read**:

```tcl
foreach f $sources { ... lappend inc_dirs [file dirname $f] }
```

A fixed path there would have compiled a swept config against some *other* config's
`DWN_TOP_LATENCY`, silently, with a wrong number as the only symptom. Deriving it means the
headers cannot come from anywhere but the RTL that was read. A second `--include-dir` argument
would have been the obvious alternative and the wrong one — two ways to specify one value.

⚠️ **One real failure, worth recording rather than just fixing.** `scripts/verify_phase1.py`
hardcoded `os.path.join(REPO, 'rtl', 'gen', 'dwn_core.v')` and crashed. Two lessons:

1. **The regression harness was the one file that broke.** Everything it guards survived the
   path change; the guard itself did not.
2. **A path grep is not sufficient.** The components were separate strings, so searching for
   `rtl/gen` could not match `os.path.join(REPO, 'rtl', 'gen', ...)`. Found only by running it.

Fixed by importing `DEFAULT_RTL_DIR` from `run_gate1` instead of respelling the path, so it now
fails loudly on the next move rather than silently reading somewhere else.

**Verified: `verify_phase1.py` 12/12** after the restructure — Gate 1 1504/1518, harness unit
tests, and all six area numbers unchanged.

### 2026-08-07 — 2a step 3: pipeline depth is an argument, not a constant

`emit_core.py` takes `--pipe-lut/--pipe-pop/--pipe-out`, `emit_encoder.py` takes `--pipe-enc`.
`PIPE_LUT/POP/OUT` survive as the *defaults*, which is what keeps `rtlgen/config.py`'s drift
guard working. Group B can now sweep depth without a code edit (dse-plan §1).

**A real trap found while doing it.** `dwn_top` instantiates the core as
`dwn_core #(.PIPE_LUT(PIPE_LUT), ...)`, so **dwn_top's own parameter defaults override the
core's**. Emitting `PIPE_OUT=0` from `emit_core.py` while `emit_encoder.py` still defaulted
`dwn_top`'s to 1 would have silently put the register back — the design would have looked
3-stage in one file and 4-stage in another, and only a timing number would have hinted at it.

Fixed by making the depth flow through **one** source rather than two: `emit_core.py` now writes
`DWN_CORE_PIPE_LUT/POP/OUT` into `dwn_core_params.vh` alongside the latency, and
`emit_encoder.py` reads them. The core's stages are deliberately **not** flags on
`emit_encoder.py` — two ways to specify one value is how they end up contradicting. This extends
the mechanism that file already existed for ("so the two emitters cannot disagree about pipeline
depth").

**Verified:**

- defaults: only `dwn_core_params.vh` changes, gaining exactly the 3 new defines at 1/1/1.
  `dwn_core.v`, `dwn_top.v`, `thermometer_encoder.v`, `dwn_top_params.vh` all byte-identical
  to committed Phase 1.
- non-default (`--pipe-out 0`): propagates to `dwn_core.v` **and** `dwn_top.v`, both reading
  `PIPE_OUT = 0`, with `DWN_CORE_LATENCY 2` / `DWN_TOP_LATENCY 3`. That 3 matches the Phase 1
  pipeline sweep's recorded 3-stage / no-OUT-reg point.
- **Gate 1: 1504/1504 core, 1518/1518 top, PASS.** Config self-test still green.

⚠️ **The non-default depth is verified textually, not behaviourally.** `run_gate1.py` still
hardcodes `rtl/gen`, so a 3-stage variant cannot be simulated yet — the golden model would need
to be told the new latency. **That is step 4**, and until it lands, no non-default pipeline
config should be trusted past "the text looks right". Do not synthesize a swept depth and quote
its Fmax before Gate 1 can run on it.

### 2026-08-04 — Encoder sharing: measured, and it is a dead end ❌

`exporter/analyze_encoder_sharing.py`. Phase 1 left one encoder question genuinely open: Vivado
is near-optimal *inside* a comparator (7.5 LUTs ≈ W/2), but nothing has ever been shared
*between* the comparators of one feature, which all compare the same value against sorted
constants. Mecik & Kumm's FloPoCo encoder presumably did exactly that, and at `md`/`lg` sizes
the encoder is what decides whether a config fits — so this is worth more than curiosity.

**Two ideas, both dead, and the reasoning is recorded so nobody re-derives them.**

**1. Binary search for the bucket index — arithmetically worse, not better.** The appeal is
obvious: 46 sorted thresholds is one question ("which of 47 buckets?"), seemingly worth
`log2 k ≈ 6` comparisons rather than 46. It does not work, because each level must *select*
which threshold to compare against next, and that is a multiplexer over 16-bit constants whose
input count doubles per level. For k=46 the level-5 mux alone is 32-to-1 over 16 bits, on the
order of 176 LUTs — more than the 345 the 46 direct comparators cost in total. Instantiating
the whole tree instead needs `1+2+4+...+32 = 63` comparators, worse than the 46 you started
with. **The mux cost is fundamental to any combinational bucket decode.**

**2. Shared high-bit prefixes — only pays if thresholds cluster, and they do not.** Measured
across every split point, with a cost model calibrated on the measured 1519/202:

| Feature | k | direct | best split | groups | shared bound |
|---|---|---|---|---|---|
| 14 (`mass_mmdt`) | 46 | 346 | 5/11 | 8 | 321 (−7%) |
| 15 (`multiplicity`) | 39 | 293 | 4/12 | 5 | 278 (−5%) |
| 0 (`zlogz`) | 37 | 278 | 4/12 | 5 | 264 (−5%) |
| 3 (`c1_b2_mmdt`) | 31 | 233 | 5/11 | 4 | 210 (−10%) |
| **total** | **202** | **1519** | | | **1441 (−5%)** |

**−5%, and that is an optimistic ceiling** — the model ignores routing and the combine's
fan-in, and was deliberately written to flatter the shared scheme. The cause is the encoding
itself: `DistributiveThermometer` places thresholds at **quantiles**, so they spread across the
data distribution and their high bits rarely agree. 46 thresholds land in 8 distinct high-bit
groups; 37 land in 5. There is nothing to share.

**Conclusion: the encoder is at its floor for this scheme.** Phase 1's *"most of the 14× is
real, not naive construction"* is now measured rather than asserted. The three levers that
remain, in order of value:

1. **`z`** — sets how many thresholds exist at all, and encoder cost saturates at
   `features × z`. A **config** change, no RTL. Still the most valuable unswept axis.
2. **Per-feature narrowing** — already measured at −17.1%, still not worth a spec change at
   `sm`, still possibly decisive at `md`/`lg`.
3. **A different encoding scheme** — `Thermometer` (evenly spaced) would cluster in high bits
   far better than `DistributiveThermometer` does, so sharing might pay there. But encoding
   scheme is a Group A axis that changes accuracy, so it is a sweep point, not an optimization.

⚠️ **Retraction, kept visible:** an earlier estimate in conversation put bucket-decode at ~2.3×
(1519 → ~670 LUTs). That was wrong — it counted `log2 k` comparators and ignored the
multiplexer cost entirely. The measured ceiling is −5%.

---

## Numbers worth quoting

*(empty — Phase 1's are in `docs/phase1-ledger.md`)*

---

## Open questions

| | |
|---|---|
| ~~Is the reduction really ~54% of the core?~~ | ✅ **Closed 2026-08-07. Yes, exactly: 50 + 58 = 108.** And one node is exactly one LUT6, measured. **But the ratio that matters is 3.6% of the whole design**, not 54% of the core — dse-plan §3's 40% bar predates the 14× encoder finding. Learnable Reduction stays deferred; see the log entry. |
| **Does `md` actually fit?** | The bound says ~80% of the part, which is tight enough that routing (not LUT count) may decide it. A failure to route is a data point, not a mistake (brief §12 risk #2). |
| **How many thresholds does a bigger model really select?** | `sm` chose 202 of 300 slots (67% unique). If that ratio holds, `md`/`lg` encoder estimates drop. Only training says. |
| **Per-feature comparator narrowing** | Measured −17.1% at `sm` and not adopted — 260 LUTs did not justify a spec change. **At `md`/`lg` it may decide whether a config fits at all.** Revisit when a config is marginal. Now the *largest* remaining RTL-side lever, since sharing is dead (below). |
| ~~Can comparators share logic across thresholds of a feature?~~ | ❌ **Closed 2026-08-04, negative.** Binary-search decode is arithmetically worse (mux cost doubles per level); shared high-bit prefixes bound at **−5%** because quantile-spaced thresholds do not cluster. See the log entry. `exporter/analyze_encoder_sharing.py` re-runs it for any checkpoint — worth repeating if a config ever uses evenly-spaced `Thermometer`, where clustering should be much stronger. |
| **`z` is the axis nobody has swept** | The paper fixes z=200 for every JSC config and never reports its cost. `z` sets the saturation ceiling on encoder area, which dominates. Accuracy vs area vs `z`, on a part where it binds, is unmeasured by anyone — probably the single most publishable axis here. |
| **Slim checkpoint format — would make every trained config committable** | Sweep checkpoints total ~933 MB and one is ~122 MB, over GitHub's per-file limit, so they are gitignored. But **91% of a checkpoint is `mapping.weights`**, an `input_bits × (nodes×n)` tensor used *only during training*: `extract_wiring()` reads it once, takes `argmax(axis=0)`, and keeps `nodes × n` integers. Storing the resolved wiring instead would cut a 101 MB checkpoint to **~350 KB** and the whole grid to **~4 MB** — committable, with faster loads as a bonus. **Not attempted mid-sweep**, deliberately: it changes the checkpoint format on the Gate 1 path, and `extract.py`'s own comment warns that `__dummy_mapping` has the same shape and dtype as a real mapping, so getting this wrong yields "a valid-looking, totally wrong export" — silently. Build it after the sweep, with a Gate 1 re-run proving slim and fat checkpoints emit byte-identical RTL. **Zipping is not the alternative: measured 1.09×**, since float32 weights are near-incompressible. |
| **3-stage pipeline** | Closes 100 MHz post-synthesis but was never re-verified post-route. One cheap Group B point. |
| **FTDI D2XX for >5 Mbaud** | Optional; the VCP driver, not the design, is the wall. Two more I/O-wall points if Phase 3 leaves time. |
| **Package the generator as a reusable tool** — *scoped, after Phase 3* | `docs/reusable-generator.md`. No open RTL implementation of DWN targets small FPGAs, which is why this project exists — and the generator is most of one already: verified across 20–2400 nodes, 1–3 layers, n=2/4/6, z=8–800, three encodings, and both wiring representations. What is JSC-specific is the *plumbing*: hardcoded Q3.12, and a harness shaped around 33-byte records and 5 classes. **~1–2 weeks, almost none of it RTL.** The load-bearing item is the MNIST port, because it is the only one that *tests* generality rather than asserting it. Split by forking after Phase 3 — the history is where the reasoning lives. |
| **MNIST port** | Stretch, after Phase 3. Scoped in `docs/phase1-ledger.md` — three harness breakages, ~1–2 days, and a *higher* accuracy number that means less, not more. |

---

## Pointers

- `docs/dse-plan.md` — what gets swept, the knob groups, the strategy
- `docs/phase2-handoff.md` — machine acceptance test + the `rtlgen/` restructure
- `docs/phase1-report.md` — what already works, and how to reproduce it
- `docs/phase1-ledger.md` — Phase 1's raw log and its open questions
- `docs/project-brief.md` §10 — the DSE study as originally specified
