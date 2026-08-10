# Phase 3 plan — the Controlled Comparison (Study 2)

**Question (brief §10):** for the same task on the same silicon, how does hand-written weightless
RTL compare to the standard FPGA-ML toolchains — and where does it sit against the wider LUT-DNN
literature?

Two halves. The hands-on half is the expensive one; the literature half is cheap and is what
actually answers the question for a reader who knows the space.

Read first: `docs/phase2-report.md` (what we are comparing), brief §8 (the literature tier),
brief §10 Study 2, brief §12 risk #6.

**Running this on a different machine?** `docs/phase3-handoff.md` first — the acceptance test,
the toolchain (Vivado 2025.2 **and an HLS-capable install** — see the ledger note on which
`vitis_hls`/`v++` command the tools expect), and the rules that keep the comparison controlled. Phase 3 is portable: everything it needs is committed, including the Phase 1
checkpoint and all 54 DWN results. No sweep checkpoints, no 166k test set, no board.

---

## 1. What Phase 2 hands over

| | |
|---|---|
| **Our best fitting config** | `1x2400 z=50` — **76.18%**, 12,751 LUTs (61.3%), **0 DSP, 0 BRAM**, 101.3 MHz, 4 cycles / 39.5 ns |
| **Best accuracy that fits** | `1x1600 z=100` — 76.35% at 66.0% |
| **Cheapest near-plateau** | `1x800 z=50` — 75.95% at **33.6%** |
| **The frontier** | 15 non-dominated points — iso-area and iso-accuracy both need a curve, not one model |
| **Noise floor** | **0.15 pp**, measured. Any accuracy difference below this is not a difference |
| **The flow** | `scripts/run_synth.py --rtl-dir` + `scripts/build.tcl` — same part, clock, threads, strategy |

That last row is what makes this *controlled* rather than a table of numbers from different
papers. Every competitor design goes through the identical flow.

---

## 2. Hands-on half

### 2.1 conifer first (a GBDT)

**Gradient-boosted**, not a plain forest — it is what brief §10 specifies, it is the standard
model class for tabular JSC classification, and it keeps the comparison apples-to-apples with
**TreeLUT** in the literature half, which also targets GBDTs.

1. Train a GBDT on JSC (xgboost or sklearn's `HistGradientBoosting`). CPU, minutes, local.
2. Sweep depth × n_estimators to get a **curve**, not one point — matching how we report DWN.
3. Emit HDL via conifer, synthesize through **our** `build.tcl` at `xc7a35tcpg236-1`, 10 ns.
   **Emit Verilog wherever the backend allows it** — see the convention note in §2.4.
4. Record the same columns we do, including BRAM/DSP.

Conifer goes first because it is the more likely of the two to synthesize cleanly at this scale
(brief §10). Bank one valid comparison before taking on hls4ml's fitting problem.

### 2.2 hls4ml second (a quantized MLP)

**The published JSC design is 63,251 LUTs against our 20,800-LUT part — it does not fit.**
Shrinking it until it does — pruning, fewer bits, smaller layers — **is the experiment**, not a
setback. "The standard flow does not fit; the weightless one uses 61%" is the result.

Expect this to be the harder of the two (brief §12 risk #6). Budget accordingly.

### 2.3 What to measure, for every design

accuracy · LUT / FF / BRAM / DSP · Fmax · latency (cycles **and** ns) · throughput · Vivado
power estimate (**flag it as an estimate**).

### 2.4 Emit Verilog, not VHDL

**Project convention: every generated HDL in this repo is Verilog.** All of `rtl/`, everything
`rtlgen/` emits, and the whole Gate 1 testbench are Verilog, and `scripts/build.tcl` reads
`.v` sources. Keeping the comparison designs in the same language means one `read_verilog` path,
one simulator invocation, and reviewers reading one language.

- **hls4ml / Vitis HLS** — emits Verilog by default. Keep it that way; do not switch to VHDL.
- **conifer's `xilinxhls` backend** — same, it goes through HLS.
- ~~⚠️ conifer's direct-to-RTL backend is VHDL by construction, so it needs a `read_vhdl` branch
  in `build.tcl`.~~ **Moot — resolved 2026-08-10 by running it.** conifer's VHDL backend cannot
  execute on Windows at all: `FixedPointConverter` shells out to a hard-coded POSIX `g++ … -fPIC
  … -o X.so` through `os.system`, and it fails earlier still on `np.random.randint(0, 2**32)`,
  which overflows Windows' 32-bit default int. **So `build.tcl` needs no `read_vhdl` branch and
  the Verilog convention holds for every row**, with no exception. See `docs/phase3-ledger.md`,
  2026-08-10.

### 2.5 Compare at iso-accuracy and iso-area

Not "our one model vs their one model." Two questions:

- **iso-accuracy** — at ~76.2%, what does each approach cost in LUTs, DSPs, latency?
- **iso-area** — given ~12,700 LUTs, what accuracy does each reach?

Our frontier supplies points at both; theirs needs a sweep too, which is why §2.1 step 2 exists.

---

## 3. Literature half

One combined table and plot placing our numbers alongside the published LUT-DNN family on JSC.
**No resynthesis** — citation and plotting, a few days, done once our own numbers are in hand.

LogicNets · PolyLUT · PolyLUT-Add · NeuraLUT · NeuraLUT-Assemble · TreeLUT · Mecik & Kumm's
thermometer-encoding DWN numbers. Include AmigoLUT / LLNN / ReducedLUT if time allows.

⚠️ **Brief §8's list is not current.** It was written early in the project. A search in August
2026 already surfaced *WARP Logic Neural Networks* (arXiv 2602.03527) and a bit-flip resilience
study of logic/LUT-based networks (arXiv 2603.22770) that are not in it. Refresh the list before
building the table — this is a fast-moving corner of the literature and a stale comparison is
the one thing a reader in this space will notice immediately.

---

## 4. ⚠️ Two comparability traps — settle these before building any table

Both are the failure mode Phase 2 kept hitting: a number that *looks* comparable and is not.

### 4.1 The literature's LUT counts are core-only

The DWN paper reports `lg` at **4,972 LUTs** — **no encoder**. Ours is 12,751 total, of which
**5,753 is encoder**. Quoting one against the other is wrong in both directions.

This is exactly why brief §6 requires core and encoder reported separately, and why our results
do. **Every row in the comparison table must state which convention it uses.** Where a paper
excludes the encoder, say so in the table, not a footnote.

### 4.2 Different silicon

| | ours | most of the literature |
|---|---|---|
| part | `xc7a35t` **-1 speed grade** | `xcvu9p` |
| clock | 100 MHz (board oscillator) | ~700 MHz reported |

**LUT counts roughly transfer. Fmax and latency-in-ns do not.** A -1 Artix-7 is a slow part by
design; the paper's speed figures are on a large, fast, speed-graded device. Report latency in
**cycles** alongside ns (brief §6) so the architectural comparison survives the part difference.

hls4ml and conifer avoid this trap entirely — they go through our flow on our part.

---

## 5. What the answer probably looks like

Published hls4ml on JSC: **76.2%, 63,251 LUTs.** Ours: **76.18%, 12,751 LUTs, 0 DSP.**

Same accuracy, ~5× fewer LUTs. That comparison currently spans two papers and two parts, so it
is a hypothesis, not a result — verifying it on our silicon is the point of §2.2.

---

## 6. Sequencing

```
conifer  ──► GBDT sweep ──► conifer HDL ──► our synthesis flow ──► table rows
                                                                      │
hls4ml   ──► baseline (will not fit) ──► shrink until it does ────────┤
                                                                      │
literature ──► pull published JSC numbers ──► combined table + plot ──┴──► Phase 3 report
```

conifer and hls4ml are deliberately **serial** — different setups, different failure modes
(brief §10). The literature half is independent of both and can start any time our own numbers
are final, which they now are — **and it needs no machine, no toolchain and no synthesis**,
which makes it the sensible thing to start with.
