# Phase 2 — DSE: running ledger

**Live document. Update it as work lands, not afterwards.** Status table first, then the
chronological log, then the numbers worth quoting, then what is still open.

**Status: not started.** Phase 1 is complete (`docs/phase1-report.md`); Gate 1b passed
166,000/166,000 on hardware.

Phase 2's question (brief §10, Study 1): *what is the accuracy/area/latency Pareto frontier for
DWN on a fixed small FPGA?* Its deliverables are Pareto plots plus one headline number — **the
largest DWN that fits an XC7A35T, and what it scores** — with **core and encoder LUTs reported
separately, always** (brief §6).

Read first: `docs/dse-plan.md` (what gets swept and why), `docs/phase2-handoff.md` (machine
setup and the restructure decision), `docs/phase1-report.md` (what already works).

---

## Status

| Step | What | Status |
|---|---|---|
| **2a** | Restructure: `rtlgen/`, per-config output under `build/`, one config drives the flow | ❌ |
| **2b** | Recalibrate the area model against measured encoder cost | ❌ |
| **2c** | Train the Group A grid (GPU-bound, Kaggle, batched) | ❌ |
| **2d** | Filter on predicted area before spending any Vivado time | ❌ |
| **2e** | Synthesize the survivors (serial Vivado, the expensive part) | ❌ |
| **2f** | Group B sweeps (pipeline depth, clock, strategy) on survivors only | ❌ |
| **2g** | Merge into one Pareto frontier + the headline number | ❌ |
| — | *Optional:* n=2 congestion characterization, reported separately | ❌ |

### Prerequisites before any of the above

| Item | Status |
|---|---|
| Machine passes `scripts\verify_phase1.py --with-board` (22/22) | ✅ on the current laptop |
| **All sweep synthesis from ONE machine and ONE Vivado version** | decide before 2e |
| Full 166k test set present (gitignored, does not travel) | ✅ here |
| `rtl/gen/` retired in favour of `build/<config>/rtl/` | part of 2a |

---

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
| `md` 1×360 | 360 | ≤ 2,160 | ≲ 16,000 | ~720 | ~80% — tight |
| `lg` 1×2400 | 2400 | → ~3,200 | ~24,000 | ~4,972 | **>100%** ❌ |

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

*(empty — first entry goes here)*

---

## Numbers worth quoting

*(empty — Phase 1's are in `docs/phase1-ledger.md`)*

---

## Open questions

| | |
|---|---|
| **Is the reduction really ~54% of the core?** | Inferred by subtraction; Vivado inlined the submodules. Synthesize `popcount`+`argmax` standalone to confirm before deciding on Learnable Reduction. |
| **Does `md` actually fit?** | The bound says ~80% of the part, which is tight enough that routing (not LUT count) may decide it. A failure to route is a data point, not a mistake (brief §12 risk #2). |
| **How many thresholds does a bigger model really select?** | `sm` chose 202 of 300 slots (67% unique). If that ratio holds, `md`/`lg` encoder estimates drop. Only training says. |
| **Per-feature comparator narrowing** | Measured −17.1% at `sm` and not adopted — 260 LUTs did not justify a spec change. **At `md`/`lg` it may decide whether a config fits at all.** Revisit when a config is marginal. |
| **`z` is the axis nobody has swept** | The paper fixes z=200 for every JSC config and never reports its cost. `z` sets the saturation ceiling on encoder area, which dominates. Accuracy vs area vs `z`, on a part where it binds, is unmeasured by anyone — probably the single most publishable axis here. |
| **3-stage pipeline** | Closes 100 MHz post-synthesis but was never re-verified post-route. One cheap Group B point. |
| **FTDI D2XX for >5 Mbaud** | Optional; the VCP driver, not the design, is the wall. Two more I/O-wall points if Phase 3 leaves time. |
| **MNIST port** | Stretch, after Phase 3. Scoped in `docs/phase1-ledger.md` — three harness breakages, ~1–2 days, and a *higher* accuracy number that means less, not more. |

---

## Pointers

- `docs/dse-plan.md` — what gets swept, the knob groups, the strategy
- `docs/phase2-handoff.md` — machine acceptance test + the `rtlgen/` restructure
- `docs/phase1-report.md` — what already works, and how to reproduce it
- `docs/phase1-ledger.md` — Phase 1's raw log and its open questions
- `docs/project-brief.md` §10 — the DSE study as originally specified
