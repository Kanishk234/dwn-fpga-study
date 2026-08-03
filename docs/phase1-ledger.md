# Phase 1 — CORE: running ledger

**Live document. Update it as work lands, not afterwards.** Status table first, then the
chronological log, then the numbers worth quoting, then what is still open.

Phase 1's exit condition is Gate 1b (brief §11): *the bitstream running on the Basys 3
reproduces the software model's JSC test-set accuracy, to the sample.* Not "it lights the right
LED for a few inputs" — the full test set through benchmark mode, matching what PyTorch
reported.

---

## Status

| Step | What | Status |
|---|---|---|
| **1a** | LUT6 mapping probe — does `TABLE[addr]` map to one LUT6? | ✅ risk #1 retired, `docs/probe-results.md` |
| **1b** | Reproduce DWN training; JSC checkpoint we trust | ✅ 74.06% vs paper's 74.0% |
| **1c** | Verilog templates, by hand, for one small real model | ✅ core + encoder |
| **1d** | Golden software model + bit-exact testbench — **GATE 1** | ✅ **both levels pass** |
| **1e** | Exporter (checkpoint → tables/wiring/thresholds) | 🟡 works, one-shot, not generalized |
| **1f** | RTL generator (export → Verilog) | ❌ `rtlgen/` empty |
| **1g** | Harness — UART, BRAM vector store, cycle counter, FSM, 7-seg | 🟡 **UART rx/tx done + tested**; BRAM store, FSM, 7-seg outstanding |
| **1h** | Board: bitstream reproduces test-set accuracy — **GATE 1b** | ❌ |

### Not in the brief's list, but Phase 1 cannot finish without them

| Item | Why it matters | Status |
|---|---|---|
| `constraints/basys3.xdc` | required for a *bitstream*; OOC synthesis does not need it | ❌ dir empty |
| `scripts/build.tcl` | non-project-mode build; the DSE sweep reuses it | ✅ |
| First synthesis run | the encoder-vs-core LUT split (brief §6) — unmeasured | ✅ **see below** |
| Pipeline registers (II=1) | brief §9; also a Phase 2 sweep axis | ✅ 4 stages, 161 MHz, II=1 |
| Full 166k test set | Gate 1b needs the whole set; we have **1000** samples | ❌ |

---

## Log

### 2026-08-02

- **1a probe** — `TABLE[addr]` maps to a single LUT6 as hoped. Risk #1 retired
  (`docs/probe-results.md`). This was the branch point for the whole area model.
- **Submodule pinned** at `9f887a0`. Upstream ships **PyTorch training only** — no RTL, no HLS,
  no FPGA flow. Confirms the gap the project exists to fill.
- **Kaggle training brought up.** `torch_dwn` has no CPU path (`EFDFunction CPU not
  Implemented`), so training runs off-machine. First build failed: upstream's `pyproject.toml`
  declares `torch` as a *build* requirement, so plain `pip install .` builds in an isolated env
  against a second, mismatched torch. Fix: `--no-build-isolation`.
- **Three training runs, three rejected hypotheses** — see `training/README.md` run log.
  z=4→8 bought +0.22pp; batch 256→32 bought +0.07pp; reading upstream showed lr/scheduler/
  epochs/tau/mapping already matched the authors' recipe exactly.
- **Checkpoint format recorded** from the pinned submodule, not inferred —
  `docs/checkpoint-format.md`. Closes risk #5.
- **Read the paper.** Table 14 gave the real JSC configs (`docs/paper-configs.md`): **z=200**
  and **single-layer**. We had been at z≤8 with a random-wired second layer.
- **Reproduced the paper's `sm` (1×50)**: 74.06% best / 73.84% final vs the paper's 74.0%.
  This is the Gate 1 reference checkpoint.
- **Python 3.12** chosen and pinned (`requirements.txt` header has the reasoning — Kaggle
  parity, and Phase 3's hls4ml/conifer lag new releases).
- **Extractor** (`exporter/extract.py`) — pulls tables/wiring/thresholds and verifies itself
  with a numpy forward pass against PyTorch's own predictions: **1000/1000**.

### 2026-08-03

- **Core RTL.** Hand-written `rtl/lut_node.v`, `popcount.v`, `argmax.v`; model-specific
  `rtl/gen/dwn_core.v` emitted by `exporter/emit_core.py`, which parses its own output back and
  checks tables, wire indices, and address bit order against the checkpoint.
- **GATE 1 PASSED (core)** — 1504/1504 via xsim. Vivado is at
  `C:\AMDDesignTools\2025.2\Vivado\bin`, not `C:\Xilinx`.
- **Precision measured, not guessed** (`exporter/analyze_precision.py`). Chose **Q3.12,
  16-bit**; **Q3.15/19-bit is the zero-bit-error fallback** if it is ever needed.
- **Gate 1 runner converted** from PowerShell to Python so `dse/` can import `run_xsim()` in
  Phase 2 instead of shelling out.
- **Thermometer encoder.** `exporter/emit_encoder.py` emits 202 comparators + `dwn_top`, with
  its own read-back check (202/202).
- **GATE 1 PASSED (top)** — 1518/1518 end to end, quantized features → encoder → core.
- **First synthesis run** (`scripts/build.tcl`, `scripts/run_synth.py`), out-of-context on
  `xc7a35tcpg236-1`. Two significant results — see *Area* and *Timing* below.
- **Pipelined.** Four stages via a parameterized `pipe_reg`, latency 4 cycles, II=1.
  81.2 → **161.0 MHz**, LUT count unchanged. Gate 1 re-run and still passing at both levels,
  now streaming a vector every clock — which is what actually proves II=1.
- **Harness started: UART.** `harness/uart_rx.v` + `harness/uart_tx.v`, 8N1, loopback-tested
  (`scripts/run_tb.py uart`): all 256 byte values back-to-back, zero mismatches, and a
  deliberately corrupted stop bit correctly reported as a framing error.
  - `BAUD` is a **parameter, defaulting to 115200**. Two reasons: first bring-up should fail
    for interesting reasons rather than marginal signal integrity, and **sweeping baud is how
    the I/O wall gets characterized** (brief §14) — so it must not be a constant. Brief §6's
    ~320 µs/sample figure assumes 1 Mbaud; raise it after Gate 1b.
  - `uart_rx` synchronizes `rx` through two flops (it is asynchronous to `clk`) and samples at
    each bit's midpoint, tolerating a half-bit of drift.
  - `uart_tx` has **no FIFO** by design: interactive mode sends one byte per classification and
    benchmark mode streams only after a run completes. `start` while `busy` is ignored, not
    queued — if a caller ever needs continuous streaming, add a FIFO around it rather than
    letting the transmitter drop bytes silently.
- **Harness: BRAM vector store + benchmark FSM.** `harness/vector_store.v`,
  `harness/benchmark_fsm.v`, tested by `scripts/run_tb.py benchmark`. Measured II=1 exactly:
  32 vectors in 37 cycles, 16 in 21, 1 in 6 — always `n + LATENCY + 1`.
  - **Found and fixed a real alignment bug.** The first version fed the "valid" flag into the
    label delay line at *issue* time but the label itself two cycles later, so every prediction
    was scored against the wrong sample's label. It produced a plausible accuracy number rather
    than an obvious failure — which is exactly why the unit test asserts an exact expected
    count with some labels deliberately wrong, instead of just checking that it runs.
  - The BRAM read and the classifier are pipelines with different entry points: a label can
    first be captured 2 cycles after its address is issued, while the prediction lands at
    `1 + LATENCY`. So the delay line is `LATENCY` deep and the valid flag has to enter it on
    the same cycle the label does.

---

## The full test set does not fit on the device

166,000 vectors × 256 bits = **42.5 Mbit**. The Basys 3 has **1.8 Mbit**. This is not a tuning
problem — Gate 1b has to run in **batches**: load `DEPTH` vectors, classify, accumulate accuracy
on-chip, repeat. `vector_store` at DEPTH=1024 costs ~265 Kbit (~15% of block RAM).

Only running totals cross the UART, never one prediction per sample. That is what `correct_count`
inside `benchmark_fsm` is for.

## UART speed ceiling

The FT2232HQ tops out at **~12 Mbaud**. At 100 MHz, `CLKS_PER_BIT = 100/12 = 8.33` — a 4% bit
error, too much for reliable 8N1. Either derive the UART clock from an MMCM that divides evenly
(96 MHz → 8, or 120 MHz → 10) or use a fractional accumulator. Clean integer rates from 100 MHz:
10M, 6.25M, 5M, 4M, 3.125M, 2M, 1M.

**Raising baud does not fix the I/O wall.** At 12 Mbaud a 32-byte sample still takes 26.7 µs
against the core's 24.8 ns — ~1,000× I/O-bound instead of ~2,600×. Streaming one sample at a
time can never reach core throughput over a serial link, which is the whole reason benchmark
mode exists (brief §9) and why quantifying this is a contribution rather than a defect (§14).

Where baud *does* matter is Gate 1b turnaround, since the full test set is 5.3 MB:

| Baud | Full test-set upload |
|---|---|
| 115,200 | ~8 min |
| 1 M | ~53 s |
| 12 M | ~4.4 s |

Plan: get Gate 1b working at 115200, then raise it — and sweeping baud *is* the I/O-wall
measurement, not a detour from it.

---

## Area — the encoder costs 14× the core

Out-of-context, post-synthesis, `xc7a35tcpg236-1`:

| Module | LUTs | % of device | FF | BRAM | DSP | comb delay |
|---|---|---|---|---|---|---|
| `dwn_core` | **108** | 0.52% | 0 | 0 | 0 | 10.194 ns |
| `thermometer_encoder` | **1519** | 7.30% | 0 | 0 | 0 | 2.962 ns |
| `dwn_top` | **1619** | 7.78% | 0 | 0 | 0 | 12.314 ns |

**The core reproduces the paper almost exactly: 108 LUTs against their reported 110** (Table 2,
`sm` 1×50). Independent confirmation that the extraction, the emitted RTL, and the area model
are all right — a different structure would not land within 2 LUTs by accident.

Zero DSPs and zero BRAM, as the paper claims for every DWN result. The model really does live
entirely in logic.

**The encoder is 14.06× the core.** Brief §12 risk #3 anticipated the encoder being
uncounted, citing Mecik & Kumm's *up to 3.2×*. Measured here it is **more than four times worse
than that worst case**. The whole design is still only 7.78% of the part, so nothing is at risk
— but every "% of Basys 3" figure derived from the paper's core-only numbers is a serious
underestimate, not a mild one.

Two caveats before this gets quoted anywhere:

1. **Our encoder is deliberately naive** — one independent 16-bit comparator per selected
   threshold, ~7.5 LUTs each. Mecik & Kumm built theirs with FloPoCo and presumably shared
   logic across thresholds of the same feature. Thermometer bits for one feature are
   comparisons of the same value against *sorted* constants, so there is real structure to
   exploit. Some of the 14× is DWN's encoder; some is ours.
2. **Post-synthesis, not post-implementation.** Routing is not done, so these are estimates.

**This makes `z` the most interesting sweep axis in Phase 2.** `z` drives the number of
selected thresholds, which drives encoder area directly — and encoder area dominates. The paper
fixes z=200 for every JSC config and never reports what it costs. Accuracy vs area vs `z`, on a
part where it binds, is unmeasured by anyone.

## Timing — pipelined, and it clears the board clock

Unpipelined the design was **12.314 ns / 81.2 MHz**, under the Basys 3's 100 MHz. Four pipeline
stages fixed that.

⚠️ **Correction (kept visible).** Earlier reasoning assumed the 202 comparators would be the
critical path. The measurement said the opposite: **core 10.194 ns, encoder 2.962 ns** — the
depth is in the popcount and argmax trees. Stages were placed accordingly.

Constrained at 10.0 ns (100 MHz), out-of-context, `xc7a35tcpg236-1`:

| Module | LUTs | % dev | FF | WNS | Fmax |
|---|---|---|---|---|---|
| `dwn_core` | 108 | 0.52% | 73 | +3.790 ns | 161.0 MHz |
| `thermometer_encoder` | 1519 | 7.30% | 0 | +7.013 ns | 334.8 MHz |
| **`dwn_top`** | **1619** | **7.78%** | **269** | **+3.790 ns** | **161.0 MHz** |

- **Pipeline: 4 stages** — after the encoder, the LUT layer, the popcounts, and the argmax.
  **Latency 4 cycles, II=1.** At the board's 100 MHz that is **40 ns** and **100 M
  classifications/s** from the core; at the achievable 161 MHz, 24.8 ns.
- **81.2 → 161.0 MHz**, and LUT count did not move at all (1619 either way). The stages cost
  flip-flops, not logic — 269 FFs on a part with 41,600.
- Stage placement is a `pipe_reg` parameter (`PIPE_ENC/LUT/POP/OUT`), not hand-wiring, because
  pipeline depth is a Phase 2 sweep axis (brief §10). `ENABLE=0` compiles a stage out entirely.

**The FF count confirms the dead-bit prediction.** `dwn_top`'s encoder register is nominally
3200 bits wide; synthesis kept **196**, having trimmed every bit that feeds no node. `dwn_core`
is exactly 73 = 50 (layer) + 20 (scores) + 3 (output), as designed.

**There is headroom worth sweeping.** +3.790 ns of slack at 100 MHz means a 3-stage version
probably still closes, trading a cycle of latency for 20 fewer FFs. That is exactly the
pipeline-depth axis Phase 2 is meant to explore, so it is a data point to collect rather than a
decision to make now.

(Still post-synthesis, pre-route. Absolute numbers will move after implementation; the
encoder-vs-core ranking will not.)

---

## Numbers worth quoting

| | |
|---|---|
| Model | JSC `sm`, 1× 50 nodes, n=6, z=200, single learnable-mapped layer |
| Accuracy | **73.84% final** (74.06% best epoch) — paper's `sm` is 74.0% |
| Core | 50 LUT6 nodes + 5× popcount(10) + argmax |
| Encoder | **202 comparators** of 3200 thermometer bits (6.3%) |
| Input format | Q3.12 signed, 16-bit → **32 bytes/sample**, matching brief §6's UART assumption |
| Q3.12 vs float32 | 10 encoder bit differences, **0 class changes** |
| Gate 1 | core 1504/1504 · top 1518/1518 |
| Area (OOC, xc7a35t) | core **108** LUTs (paper: 110) · encoder **1519** · top **1619** = 7.78% |
| Encoder / core | **14.06×** — vs the 3.2× worst case brief §12 risk #3 anticipated |
| Flip-flops | 269 total (196 encoder + 50 layer + 20 scores + 3 out) of 41,600 |
| Fmax | **161.0 MHz** pipelined (+3.790 ns slack at 100 MHz); 81.2 MHz unpipelined |
| Latency | **4 cycles**, II=1 → 40 ns at the board's 100 MHz, 24.8 ns at 161 MHz |
| Throughput | **100 M classifications/s** core-side at 100 MHz — but I/O-bound in practice (brief §6) |

⚠️ **Quote 73.84%, not 74.06%.** The saved weights are the final epoch; there is no
best-checkpoint tracking. The `.npz` predictions come from those same weights, so Gate 1 is
unaffected — but the number that matches the shipped artifact is the final one.

### Learnable Mapping did feature selection for free

Comparators per input feature, of 202:

| Feature | Comparators | | Feature | Comparators |
|---|---|---|---|---|
| `mass_mmdt` | 46 | | `m2_b2_mmdt` | 6 |
| `multiplicity` | 39 | | `c1_b0_mmdt` / `c2_b2_mmdt` / `d2_b1_mmdt` / `d2_a1_b1_mmdt` | 3 each |
| `zlogz` | 37 | | `c2_b1_mmdt` / `m2_b1_mmdt` | 2 each |
| `c1_b2_mmdt` | 31 | | `d2_a1_b2_mmdt` / `n2_b2_mmdt` | 1 each |
| `c1_b1_mmdt` | 17 | | **`d2_b2_mmdt`** | **0 — never read** |
| `n2_b1_mmdt` | 8 | | | |

Four features carry 153 of 202 comparisons, and the ones it favours (jet mass, particle
multiplicity) are the physically obvious discriminators — evidence the model learned structure
rather than noise. `d2_b2_mmdt` affects no output at all, so the UART protocol could ship 15
features instead of 16 (~6% off an I/O-bound path). **Don't act on that yet** — a different
seed would likely drop a different feature; confirm it is stable across configs first.

---

## Open questions and risks

| | |
|---|---|
| **Is our encoder unnecessarily large?** | 14.06× the core, vs Mecik & Kumm's 3.2×. One 16-bit comparator per threshold is the naive construction; thresholds of one feature are sorted, so shared logic should be possible. Worth one optimization attempt before treating 14× as *DWN's* encoder cost rather than *ours*. |
| **Post-synth, not post-implementation** | Area and timing are pre-route estimates. Needs `place_design`/`route_design` for numbers that can be published. |
| **Only 1000 test samples are local** | The full 166k set is on Kaggle. Gate 1b requires all of it, and the Q3.12 "0 class changes" result is 1000-sample evidence, not proof. |
| **Nothing has touched silicon** | Gate 1 is simulation. Brief §12 risk #7: UART framing, BRAM addressing, reset sequencing and timing closure are all untested. |
| **No pipelining, and it is now required** | 81.2 MHz < the board's 100 MHz. Stages belong in the core (10.194 ns), not the encoder (2.962 ns). |
| **Exporter is one-shot** | `emit_core.py` / `emit_encoder.py` target one model. Phase 2 needs `rtlgen/` to generalize over the sweep grid. |
| **`rtl/gen/` is committed — decide before the first sweep run** | Fine now: one model, and these are the exact files Gate 1 verified and synthesis measured, so history is useful. It does not survive 40–70 sweep configs. Undoing it later is cheap (`git rm --cached rtl/gen/` + a `.gitignore` line, no history rewrite), so the deadline is **before the sweep starts committing configs**, not the start of Phase 2 as such. |

---

## Pointers

- `docs/project-brief.md` — the full plan; §6 resource budgets, §11 phase breakdown, §12 risks
- `docs/checkpoint-format.md` — what the exporter reads, verified against the pinned submodule
- `docs/paper-configs.md` — the paper's JSC configs (Table 14 / Table 2) and what they changed
- `docs/probe-results.md` — Phase 1a, risk #1 evidence
- `training/README.md` — the run log and how to retrain
