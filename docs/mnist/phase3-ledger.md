# MNIST Phase 3 ledger — the Controlled Comparison on a second dataset

Running log for MNIST Phase 3. Phase 2 is closed: `docs/mnist/phase2-report.md`, log in
`docs/mnist/phase2-ledger.md`. The JSC equivalents are `docs/phase3-plan.md`,
`docs/phase3-ledger.md` and `docs/phase3-report.md` — **read those first; most of the method is
already settled and should be reused rather than re-derived.**

**Split, as agreed:** the *literature half* (3L-\*) needs no board, no Vivado and no synthesis. The
*hands-on half* (3M-\*) needs the Phase 1/2 toolchain — Vivado 2025.2 at `xc7a35tcpg236-1`, 10 ns.
Both halves log here. Where an entry corrects one written from the other side, say so rather than
editing it away.

**Correct entries rather than appending to them.** Phase 2 withdrew four claims this way, including
its own headline. A wrong turn left visible is worth more than a tidy log.

---

## Status

| Step | What | Who | Status |
|---|---|---|---|
| 3L-a | Refresh the MNIST literature list | literature | ⬜ |
| 3L-b | Confirm the encoder convention applies unchanged | literature | ⬜ — likely a re-check, not new work |
| 3L-c | Per-paper MNIST numbers into a machine-readable table | literature | ⬜ `cc/literature/mnist_literature.json` |
| 3L-d | Combined table + Pareto plot against our frontier | literature | ⬜ |
| 3L-e | Phase 3 report | literature | ⬜ `docs/mnist/phase3-report.md` |
| 3M-a | conifer (GBDT) on MNIST | hands-on | ⬜ |
| 3M-b | hls4ml (quantized MLP) on MNIST | hands-on | ⬜ |

---

## 1. What Phase 2 hands over

Every number below is measured, place-and-routed at `xc7a35tcpg236-1` / 10 ns, and snapshotted in
`docs/results-mnist/sweep-results.json`. **25 configurations, 0 failures, Gate 1 25/25 bit-exact,
Gate 1b 10,000/10,000 on silicon.**

The rows a comparison table should quote:

| role | config | acc% | LUTs | %dev | Fmax |
|---|---|---|---|---|---|
| **best that meets 100 MHz** | `2x[1000,500]` (the paper's) | 97.76 | 3,464 | 16.65 | **103.8** |
| best small, meets clock | `1x500` | 97.70 | 2,246 | 10.80 | 108.4 |
| cheapest usable | `1x300` | 96.77 | 1,597 | 7.68 | 107.5 |
| highest accuracy | `2x[2000,1000]` | 98.32 | 5,670 | 27.26 | 92.3 ✗ |
| best accuracy/area | `1x2000` | 98.26 | 6,294 | 30.26 | 94.0 ✗ |

⚠️ **The two highest-accuracy rows MISS the board clock.** Any table that lists them without
saying so is comparing a design that runs against designs that do not. JSC's Phase 3 quoted
frontier points at the board clock; do the same here.

---

## 2. ⚠️ The traps — read before building any table

### 2.1 The noise floor is 0.24 pp, and it is larger than most gaps in the MNIST literature

`docs/mnist/reduction-ledger.md`, measured over four configurations × four seeds. It already
withdrew three of this project's own claims, **including the study's headline** — the taper that
appeared to be the best model lost by 0.10 pp when averaged over seeds.

Published MNIST accuracies cluster tightly (PolyLUT 96%, NeuraLUT 96%, ULEEN and BTHOWeN in the
same region). **Many pairwise gaps in that literature are smaller than 0.24 pp.** A ranking built
on them is a ranking of seeds.

**Rule for this phase: never rank two designs on an accuracy gap below 0.24 pp.** Say they are
indistinguishable and rank on area or timing instead, which are exact.

Also from that measurement: **the same seed does not reproduce**, by up to 0.17 pp on a rerun
(GPU non-determinism). So **quote no MNIST accuracy of ours to more than one decimal.**

### 2.2 The encoder-convention trap carries over unchanged

`docs/phase3-plan.md` §4.1 and `REPORT.md` §5.2. Published LUT counts are frequently **core-only**,
excluding the input encoder; ours are encoder-inclusive, which is the stricter convention.

For MNIST this matters *less* than for JSC but is still real: the encoder is **72.3%** of `1x100`
and **20.9%** of `1x2000`, against JSC's ~42% at comparable width. So the correction is
width-dependent here, not a single factor — state it per row.

`docs/results-mnist/sweep-results.json` carries `dwn_core_luts` and `thermometer_encoder_luts`
separately for exactly this reason. **Report both conventions, as Phase 3 did for JSC.**

### 2.3 ✅ The dataset-ambiguity trap does NOT apply — and that is worth saying out loud

JSC Phase 3's largest correction was that **"JSC" is two different datasets ~1.05 pp apart**
(`docs/phase3-ledger.md`, 2026-08-10), and the standard comparison table conflated them.

**MNIST has one canonical split**: `mnist_784` ships in train-then-test order and the last 10,000
rows are the test set every published number uses. `datasets.MNIST.test_split = 'tail:10000'`
encodes it, and `scripts/dump_testset.py` refuses any dataset whose split is not reproducible.

So this entire class of correction is absent. **Do not import JSC's dataset caveat into the MNIST
table** — it would be a real error to carry a warning that does not apply.

### 2.4 The comparison DWN has been waiting for

**ULEEN and BTHOWeN are weightless neural networks — the same lineage as DWN — and they benchmark
on MNIST while reporting no JSC at all** (`docs/mnist/phase1-ledger.md`). JSC Phase 3 could only
compare DWN against BDTs, quantized MLPs and LUT-DNNs; on MNIST it can be compared against its own
family for the first time.

⚠️ Check what "LUTs" means for each of them before tabulating. Weightless models are
lookup-table-based by construction, so a LUT count may be a *model size* rather than an FPGA
resource count. Getting that wrong would produce the most embarrassing row in the table.

---

## 3. Hands-on half — what is already solved

**Do not re-derive the toolchain.** `docs/phase3-ledger.md` (2026-08-10) records the conifer and
hls4ml flows working at Vivado 2025.2, plus five silent failures already found and fixed. The ones
most likely to recur:

- **A NaN base score** in the conifer export, which would have poisoned every row.
- **Emit Verilog, not VHDL** (plan §2.4) — a project convention, and it keeps the flow uniform.
- Two silent hls4ml failures, one of which nearly shipped.

**What is genuinely new for MNIST, and both are worth predicting before measuring:**

- **conifer on 784 raw pixels.** A GBDT did not reach DWN's accuracy on JSC's 16 features. MNIST
  has 784 and no feature engineering, which is unfavourable for trees — but that is a prediction,
  not a result, and a surprise here would be a genuine finding.
- **hls4ml at 784 inputs.** On JSC it fit only at quarter width. The first layer alone scales with
  input count, so expect it not to fit, and **report a failure to fit as a measurement** — brief
  §12 risk #2, the same rule the sweep uses.

Measure the same things Phase 3 measured (plan §2.3), and report latency in **cycles** as well as
ns — the board clock does not transfer between designs.

---

## 4. Sequencing

The two halves are independent and neither blocks the other. The literature half can start now and
needs nothing from the toolchain.

Suggested order for the literature half, matching what worked for JSC:

1. **3L-a** — build the paper list first, before any numbers. JSC's brief §8 list was stale.
2. **3L-b** — for each paper, record *which convention* its LUT count uses. Do this while reading,
   not afterwards; it is unrecoverable later without re-reading everything.
3. **3L-c** — machine-readable table. `cc/literature/jsc_literature.json` is the schema to copy.
4. **3L-d** — plot against our frontier, **at the board clock**, both conventions.
5. **3L-e** — report.

---

## Log

*(empty — Phase 3 has not started. Add a dated entry per work session, newest first. Mark each
entry `[literature]` or `[hands-on]` so the two halves stay attributable.)*
