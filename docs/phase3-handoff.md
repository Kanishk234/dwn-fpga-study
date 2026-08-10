# Handoff: running Phase 3 on a different machine

Phase 3 is the Controlled Comparison (`docs/phase3-plan.md`). This covers what a second machine
needs before starting, what to check while developing, and the rules that keep the comparison
*controlled* rather than a table of numbers from different flows.

**Good news up front: Phase 3 is far more portable than Phase 2 was.** Everything it needs is in
git — the Phase 1 checkpoint, its test vectors, and all 54 measured DWN results. No 933 MB of
sweep checkpoints, no 166k test set, no board.

---

## Part 1 — Before writing any code

### 1.1 The acceptance test is not a formality

Phase 3 compares hls4ml and conifer designs **against our DWN numbers**, which were measured on
the original machine with Vivado 2025.2. If this machine synthesizes differently, the comparison
is between two flows and the word "controlled" stops being true.

```bat
git clone <repo> && cd dwn-fpga
git submodule update --init third_party/DWN
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

.venv\Scripts\python.exe scripts\verify_phase1.py
```

**Expect `12/12 checks passed`.** No board needed and no `--with-board` — Phase 3 measures area
and timing from Vivado reports, so Gate 1b is not in scope.

| Must match exactly | Value |
|---|---|
| Gate 1 vectors | 1504 core, 1518 top |
| LUT nodes / comparators | 50 / 202 |
| `dwn_core` | 108 LUTs, 73 FF |
| `thermometer_encoder` | 1519 LUTs, 0 FF |
| `dwn_top` | 1619 LUTs, 269 FF |

Timing slack may drift by tens of picoseconds (placement is stochastic). **LUT and FF counts may
not.**

**Same Vivado version is necessary but not sufficient.** Patch level can differ within a
release, and the emitted RTL is produced by *Python*, so a different `numpy` could in principle
change the tables — that is what the pins in `requirements.txt` exist for, and this test is what
confirms they held. 15 minutes, and it is the only evidence the whole chain is wired correctly.

**If LUT or FF counts differ:** stop. Either match the toolchain, or re-measure the DWN side on
this machine and say so in the writeup. Do not mix them in one table.

### 1.2 Toolchain

| | why | check |
|---|---|---|
| **Vivado 2025.2** | must match the DWN measurements | `verify_phase1.py` |
| **Vitis HLS** | hls4ml compiles to HLS C++ first — Vivado alone is not enough | confirm **before** starting §2.2 of the plan; it is a large separate install and a bad thing to discover late |
| **conifer** | `pip install conifer` | build a trivial 2-tree model and synthesize it before touching JSC |
| **hls4ml** | `pip install hls4ml` | same — smallest possible model through the flow first |

Both lag new Python releases, which is the reason `requirements.txt` pins 3.12. If they refuse
to install on this machine's interpreter, that is the cause.

⚠️ **Do not add hls4ml or conifer to `requirements.txt`.** That file is the environment
`verify_phase1.py` validates; adding heavy, version-fragile packages to it risks breaking the
reproducibility check for everyone. Use a separate `cc/requirements-cc.txt`.

### 1.3 What travels, and what does not

**Travels (all committed):**
- `training/artifacts/dwn_jsc_t200_distributive_50_l_b100_*` — the Phase 1 reference, which is
  all `verify_phase1.py` needs
- `docs/results/` — all 54 measured DWN configs, both figures, the trained grid

**Does not travel, and Phase 3 does not need it:**
- `training/artifacts/sweeps/` — ~933 MB of sweep checkpoints, gitignored. Only needed to
  *re-emit* a DWN config's RTL. The measured numbers are in `docs/results/`.
- `training/artifacts/*_testset_full.npz` — the 166k set, gitignored, `--with-board` only
- A Basys 3 — not required for Phase 3

If a specific DWN config ever needs re-synthesizing for a like-for-like comparison, copy that one
checkpoint across rather than the whole folder.

---

## Part 2 — While developing

### 2.1 One flow, always

Every competitor design goes through **our** synthesis path, not the vendor's default project
flow:

```bat
.venv\Scripts\python.exe scripts\run_synth.py --rtl-dir cc\conifer\rtl --impl
```

Same part (`xc7a35tcpg236-1`), same 10.0 ns constraint, same `general.maxThreads 8`, same
out-of-context mode. `scripts/build.tcl` takes sources as arguments and derives include paths
from them, so pointing it at generated HDL requires no edits.

**Use `--impl`.** Post-synthesis timing uses estimated routing and is systematically optimistic
— Phase 1 measured 161.0 MHz post-synthesis against 147.1 post-route on the same design.

### 2.2 Record the same columns we do

accuracy · **LUT / FF / BRAM / DSP** · Fmax · latency (cycles **and** ns) · throughput · Vivado
power estimate, flagged as an estimate.

**BRAM and DSP are not optional.** Every DWN config measured **0 / 0** across 52 designs — that
is the central claim against hls4ml, whose quantized MLPs spend DSPs on multiply-accumulate. A
table without those columns cannot make the point.

### 2.3 Sweep, do not point-compare

The plan asks for iso-accuracy **and** iso-area comparisons. Both need a *curve* from each side,
not one model each. Our frontier has 15 non-dominated points; conifer needs a depth ×
n_estimators sweep to match, and hls4ml needs its shrink sequence recorded as points rather than
only its endpoint.

### 2.4 The 0.15 pp rule

Run-to-run training noise was measured at **0.15 pp** (same config, same seed, two sessions:
73.8361% vs 73.9855%). **Any accuracy difference below that is not a difference.** This killed
two apparent Phase 2 findings; do not let it manufacture a Phase 3 one.

### 2.5 Keep a ledger

`docs/phase3-ledger.md`, updated **in the same pass as the work** (CLAUDE.md). Status table,
dated log, numbers worth quoting, open questions. When a result overturns an earlier conclusion,
**correct it and say what was retracted** — Phase 2's most useful entries are the ones recording
what turned out to be wrong.

### 2.6 Where things go

```
cc/conifer/     the GBDT, its conifer project, generated HDL
cc/hls4ml/      the MLP, its hls4ml project, generated HDL
cc/requirements-cc.txt
build/cc/       all synthesis output -- gitignored, like everything under build/
docs/results-cc/  the committed comparison numbers, via a snapshot like dse/report.py --snapshot
```

Generated HDL is a build product. It belongs in `build/`, not in git — the same rule that
retired `rtl/gen/` in Phase 2.

---

## Part 3 — Two traps that will corrupt the table

Both are the Phase 2 failure mode: a number that *looks* comparable and is not.

**1. The literature's LUT counts are core-only.** The DWN paper reports `lg` at 4,972 LUTs with
**no encoder**. Ours is 12,751 total, of which 5,753 is encoder. Every row in the comparison
table must state which convention it uses — in the table, not a footnote.

**2. Different silicon.** Published numbers are `xcvu9p` at ~700 MHz; ours is `xc7a35t` **-1
speed grade** at 100 MHz. LUT counts roughly transfer; Fmax and latency-in-ns do not. Report
latency in **cycles** alongside ns so the architectural comparison survives the part difference.

hls4ml and conifer avoid trap 2 entirely, because they go through our flow on our part. Trap 1
applies to them too — report core and encoder separately for DWN in every table.

---

## Part 4 — What can start with no machine at all

**The literature half.** Pulling published JSC numbers into a combined table and plot needs no
toolchain, no GPU, no synthesis — only our final numbers, which are committed in
`docs/results/`. It is the highest value-per-hour item in Phase 3 and has zero setup cost.

⚠️ **Check for newer entries than brief §8's list.** It was written earlier in the project; a
search in August 2026 already turned up *WARP Logic Neural Networks* (arXiv 2602.03527) and a
bit-flip resilience study of logic/LUT networks (arXiv 2603.22770) that are not in it.
