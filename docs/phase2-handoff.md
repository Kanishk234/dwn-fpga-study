# Handoff: moving to a new machine, and starting Phase 2

Phase 1 is complete and frozen. This document covers two things, in order:

1. **Prove Phase 1 reproduces here first.** Do not start Phase 2 work until it does.
2. **The restructure Phase 2 should do**, decided in advance so it isn't rediscovered mid-sweep.

Background reading, in this order: `docs/phase1-report.md` (what was built and what broke),
`docs/dse-plan.md` (what Phase 2 sweeps), `docs/project-brief.md` §10 (the DSE study).

---

## Part 1 — Acceptance test

### Why this is not a formality

**A Pareto frontier assembled from two Vivado versions is two half-frontiers with an unknown
offset between them.** Vivado's optimizer changes between releases, so identical RTL can land on
different LUT counts. If this machine does not reproduce Phase 1's areas *exactly*, its sweep
points cannot appear in the same table as Phase 1's numbers.

Finding that out on day one costs an afternoon. Finding it out at sweep point 40 costs the
sweep.

### Setup

Full cold-start instructions are `docs/phase1-report.md` §5 — toolchain, Python, Vivado, the
Kaggle test-set dump. In short:

```bat
git clone https://github.com/Kanishk234/dwn-fpga.git
cd dwn-fpga
git submodule update --init third_party/DWN
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Requirements: **Python 3.12** (not 3.14 — `torch==2.13.0` may have no wheel; not 3.10 —
`numpy>=2.3` needs ≥3.11), **Vivado 2025.2**, and a Basys 3 for the hardware half.

### Run it

```bat
.venv\Scripts\python.exe scripts\verify_phase1.py
.venv\Scripts\python.exe scripts\verify_phase1.py --with-board
```

The first form is simulation + synthesis and needs no board (~10 min). The second adds the
bitstream, programming and Gate 1b.

Expect `PHASE 1 REPRODUCES ON THIS MACHINE -- safe to start Phase 2 here.`

### What must match exactly, and what may drift

| Must match exactly | Value |
|---|---|
| Gate 1 vectors | 1504 core, 1518 top |
| LUT nodes / comparators | 50 / 202 |
| `dwn_core` | 108 LUTs, 73 FF |
| `thermometer_encoder` | 1519 LUTs, 0 FF |
| `dwn_top` | 1619 LUTs, 269 FF |
| board design | 2058 LUTs, 865 FF, 8 BRAM, 0 DSP |
| Gate 1b | 166000/166000, 166815 cycles |
| accuracy | 73.8361% float32, 73.8349% Q3.12 |

| Allowed to vary | Why |
|---|---|
| timing slack (±tens of ps) | placement is stochastic; only the sign matters |
| wall-clock times | host speed, USB, FTDI latency timer |
| link rate | same |

**If LUT or FF counts differ**, the toolchain differs. That is not a bug — but either match the
Vivado version, or re-measure Phase 1 here and say so in the writeup. Do not mix them in one
table.

### Things that do not travel with a clone

- **The full 166k test set** (`training/artifacts/*_testset_full.npz`) — gitignored, 9.4 MB.
  Copy it across, or regenerate with `training/dump_testset_kaggle.ipynb` (**inference only** —
  re-running the *training* notebook produces a different model and invalidates every number
  above). Gate 1b is skipped without it.
- **The FTDI latency timer** — a per-machine driver setting. 16 ms by default costs ~21% of a
  run with no other symptom. `scripts\host.py` detects and warns;
  `--port COMn --set-latency 1` fixes it from an Administrator shell, then replug.
- **`CLAUDE.md`** if it is still gitignored — check `git ls-files CLAUDE.md` returns something.

### Fastest possible smoke test

Before trusting any new build, flash the frozen artifact and confirm the board, cable and driver
are fine:

```bat
.venv\Scripts\python.exe scripts\program.py --bit releases\phase1\dwn_basys3_top.bit
.venv\Scripts\python.exe scripts\host.py --gate1b
```

166000/166000 means everything outside your changes works. That is also the first thing to try
whenever a Phase 2 change makes the board misbehave — it separates "the board" from "my change"
in a minute rather than an afternoon.

---

## Part 2 — The restructure Phase 2 should do

### `rtlgen/` vs `rtl/gen/` — the naming is currently wrong

| | Meaning | State today |
|---|---|---|
| `rtlgen/` | the **tool** that writes Verilog | empty — the code is in `exporter/` |
| `rtl/gen/` | the **Verilog it wrote** | committed Phase 1 output |

Brief §11 draws a real distinction worth keeping: **exporter** = checkpoint → tables, wiring,
thresholds (`extract.py`); **rtlgen** = that export → Verilog (`emit_core.py`,
`emit_encoder.py`). Phase 2 will want them separable — estimating area from a checkpoint without
emitting RTL is exactly the filtering step `docs/dse-plan.md` calls for.

**Decision, to be done while generalizing the emitters rather than as separate churn:**

```
exporter/               extract.py -- checkpoint -> tables, wiring, thresholds
rtlgen/                 emit_core.py, emit_encoder.py -- export -> Verilog   (moved here)
build/<config>/rtl/     what they emitted for that config   (gitignored)
rtl/                    hand-written primitives, unchanged: lut_node, popcount, argmax, pipe_reg
```

**`rtl/gen/` goes away.** Once the sweep produces 40–70 configs, generated Verilog is per-config
output, and CLAUDE.md's own rule already says *"`build/` is the output root… nothing else in the
repo should accumulate build products."* Generated RTL is a build product.

This also settles the deferred question of whether to gitignore `rtl/gen/`: it doesn't get
gitignored, it stops existing. The Phase 1 copy is already preserved twice over — by the
`phase1-complete` tag and by `releases/phase1/`.

`rtl/` itself stays as source: every config instantiates the same hand-written primitives.

### What generalizing actually means

The emitters already handle arbitrary layer counts, widths, `n` and class counts — `extract.py`
reads all of it from the checkpoint and `emit_core.py` loops over layers. What they do **not**
do is take a config as an argument and write somewhere other than `rtl/gen/`. Phase 2 needs:

- output path as a parameter, not a constant
- pipeline depth as a config field, not the module defaults
- a config → (RTL, area estimate, synthesis result) pipeline the sweep can call in a loop
- `dse/` on top of that, holding the grid, the runner and the Pareto plotting

`scripts/run_synth.py` already exposes `run_one()` and `scripts/run_gate1.py` exposes
`run_xsim()`/`find_vivado_bin()` for exactly this — the sweep should import them, not shell out
and parse stdout.

### Two Phase 1 findings that shape the grid

**The encoder decides what fits, not the model.** Comparator count grows with node count and
saturates at `features × z` = 3200. The paper's `lg` (2400 nodes) would need ~24,000 LUTs of
encoder on a **20,800-LUT** device against a 4,972-LUT core — so `lg` almost certainly does not
fit, and the network is not the reason. Budget sweep points accordingly; a config that fails to
fit is a data point, not a wasted run.

**`z` is the most interesting axis and the paper never swept it.** It fixes z=200 for every JSC
config and never reports its cost. `z` drives the number of selected thresholds, which drives
encoder area, which dominates. Accuracy vs area vs `z` on a part where it binds is unmeasured by
anyone.

Also queued in the ledger's open questions: per-feature comparator narrowing (measured −17.1%,
not worth adopting at `sm` size, likely decisive at `md`/`lg`), the 3-stage pipeline point, and
FTDI D2XX for >5 Mbaud.

### Before the first sweep run

`docs/phase1-ledger.md` records one decision that must be made **before** the sweep starts
committing per-config artifacts, not after: what gets committed per config. The restructure
above answers it — code in `rtlgen/`, output in `build/`, nothing per-config in git.
