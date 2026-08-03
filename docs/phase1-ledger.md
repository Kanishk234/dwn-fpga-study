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
| **1g** | Harness — UART, BRAM vector store, cycle counter, FSM, 7-seg | ❌ `harness/` empty |
| **1h** | Board: bitstream reproduces test-set accuracy — **GATE 1b** | ❌ |

### Not in the brief's list, but Phase 1 cannot finish without them

| Item | Why it matters | Status |
|---|---|---|
| `constraints/basys3.xdc` | required for any synthesis targeting the board | ❌ dir empty |
| `scripts/build.tcl` | non-project-mode build; the DSE sweep reuses it | ❌ |
| First synthesis run | the encoder-vs-core LUT split (brief §6) — unmeasured | ❌ |
| Pipeline registers (II=1) | brief §9; also a Phase 2 sweep axis | ❌ combinational |
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
| **Encoder area is unmeasured** | 202 comparators against a 50-LUT core. Brief §6 says report core and encoder separately, always; nobody has measured this where it binds. Needs the first synthesis run. |
| **Only 1000 test samples are local** | The full 166k set is on Kaggle. Gate 1b requires all of it, and the Q3.12 "0 class changes" result is 1000-sample evidence, not proof. |
| **Nothing has touched silicon** | Gate 1 is simulation. Brief §12 risk #7: UART framing, BRAM addressing, reset sequencing and timing closure are all untested. |
| **No pipelining** | Combinational end to end. Register placement should follow the first synthesis timing report, not precede it — the comparators are the likely critical path. |
| **Exporter is one-shot** | `emit_core.py` / `emit_encoder.py` target one model. Phase 2 needs `rtlgen/` to generalize over the sweep grid. |

---

## Pointers

- `docs/project-brief.md` — the full plan; §6 resource budgets, §11 phase breakdown, §12 risks
- `docs/checkpoint-format.md` — what the exporter reads, verified against the pinned submodule
- `docs/paper-configs.md` — the paper's JSC configs (Table 14 / Table 2) and what they changed
- `docs/probe-results.md` — Phase 1a, risk #1 evidence
- `training/README.md` — the run log and how to retrain
