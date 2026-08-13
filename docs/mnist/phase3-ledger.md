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
| 3L-a | Refresh the MNIST literature list | literature | ✅ **done 2026-08-13** — 9 papers + a dedicated survey |
| 3L-b | Confirm the encoder convention applies unchanged | literature | ✅ **done 2026-08-13 — it applies, and it is the biggest single correction.** Reading DWN's core-only rows against our encoder-inclusive ones would have shown us 2.3x too large |
| 3L-c | Per-paper MNIST numbers into a machine-readable table | literature | ✅ **done 2026-08-13 — 26 rows, 21 verified** in `cc/literature/mnist_literature.json`. 4 low-priority items left in `pending` |
| 3L-d | Combined table + Pareto plot against our frontier | literature | ✅ **done 2026-08-13** — `table.py --benchmark mnist`, `plot.py --benchmark mnist --snapshot`; figure in `docs/results-cc-mnist/` |
| 3L-e | Phase 3 report | literature | ⬜ `docs/mnist/phase3-report.md` |
| 3M-a | conifer (GBDT) on MNIST | hands-on | 🏁 **CLOSED 2026-08-13 at two measured rows** → `docs/results-cc-mnist/`. Iso-area: **DWN +13.45 pp at matched LUTs**; conifer **4.6× faster**. ⚠️ no curve, no measured ceiling |
| 3M-b | hls4ml (quantized MLP) on MNIST | hands-on | ❌ **CUT 2026-08-13, deliberately.** A published MNIST row exists, it will not fit 784 inputs on this part, and JSC measured this axis in full. Reasons in the log |

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

`docs/phase3-plan.md` §4.1 and `docs/jsc-report.md` §5.2. Published LUT counts are frequently **core-only**,
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

### 2026-08-13 — [literature] 3L-d: `table.py` and `plot.py` are benchmark-aware; JSC byte-identical

Both scripts take `--benchmark {jsc,mnist}` and read their inputs from a `BENCHMARKS` table rather
than JSC literals. **JSC output verified byte-identical** for both, at every step. Figure:
`docs/results-cc-mnist/mnist.png`.

#### ⚠️ Two real defects found by pointing the tools at a second dataset

**1. `fits` meant "met the clock it was constrained at", not "runs on our board."** A Group B config
built at a 12 ns constraint reports `meets_timing: true` at **87 MHz** — and was being listed in the
comparison table as a design that works. JSC never exposed this because its 12 ns variant still
reached 102 MHz. Now `fits` also requires `fmax >= 100 MHz`. **This is the exact error §1 of this
ledger warns about**, and it was in the tool rather than the table.

**2. `--dataset all` would have treated a note as a dataset.** `jsc_literature.json`'s `datasets`
dict carries an `offset_pp` entry describing the 1.05 pp gap between the two JSC datasets — not a
dataset. Introduced by me when generalising; fixed by intersecting against the datasets that
actually appear in rows.

Also JSC-specific and now conditional: the `offset_pp` note (MNIST has no variants), the
`jsc-<name>.png` filename, the `JSC-<NAME>` panel heading, and the closing "two figures, never one"
message — which for MNIST becomes *one* figure, and says so because the single canonical split is
the **absence** of JSC's defect rather than an oversight.

⚠️ **A factual error in the figure, caught on inspection:** the subtitle read "published (xcvu9p)",
but MNIST's BTHOWeN rows are **xc7z020** — the same 7-series family and `-1` speed grade as ours.
Now reads `xcvu9p + xc7z020-1`.

#### What the figure shows

Our curve sits at the far left, second only to the DWN paper's own hollow (core-only) markers, with
every other method one to two orders of magnitude to the right. **Only our designs, the DWN paper's,
and BTHOWeN-Small fall left of the 20,800-LUT device line** — i.e. almost nothing in the published
MNIST literature would fit on a Basys 3 at all.

Marker fill carries the accounting convention (filled = encoder included, hollow = core only,
square = no separate encoder), which is what stops the figure from making the comparison error §2.2
warns about.


### 2026-08-13 — [literature] 3L-a/3L-c substantially complete: 26 rows, 21 verified

PolyLUT-Add read directly — **third independent confirmation** that PolyLUT HDR is 96% / 70,673
(its own paper, NeuraLUT, and PolyLUT-Add all print it identically). PolyLUT-Add itself reaches the
same accuracy at **14,810 LUTs, a 4.8× reduction**, the best LUT-DNN area in the table.

⚠️ **The 97.5% / 75,131 figure appears in none of the three.** Its probable source is the **2025
extended PolyLUT paper** (arXiv:2501.08043, "…Hardware-Aware Structured Pruning") — a later,
different configuration. Recorded as an *additional* row to find, not a correction. Do not
attribute it to the 2023 PolyLUT.

#### A dedicated survey exists, and it is accurate

`arXiv:2506.07367`, "A Survey on LUT-based Deep Neural Networks Implemented in FPGAs". Its Table I
is the most complete MNIST comparison found. **Its rows for DWN (692/422, 1,413/1,143, 4,082/3,385),
PolyLUT (70,673/4,681), NeuraLUT (54,798/3,757) and PolyLUT-Add (14,810/2,609) match the primaries
we read directly, exactly** — so it transcribes reliably, and its other rows can be trusted enough
to record at `confidence: reported`:

| | acc | LUT | Fmax | latency |
|---|---|---|---|---|
| **TreeLUT (I)** | 97% | **4,478** | 791 | 2.5 ns |
| TreeLUT (II) | 96% | 3,499 | 874 | 2.3 ns |
| AmigoLUT-LogicNet-XS ×2 | 94.7% | 9,711 | 569 | 12.3 ns |
| AmigoLUT-NeuraLUT ×4 | 95.5% | 16,081 | 925 | 7.6 ns |
| ReducedLUT (HDR-5L) | 95.7% | 47,484 | 295 | 17 ns |

**TreeLUT is the closest competitor to DWN on area** — a GBDT, LUT-only with no BRAM or DSP, at 97%
in 4,478 LUTs. DWN `sm` still beats it 6.5× (692 LUTs at 97.1%). This is also the direct
counterpart to Phase 3's conifer work on JSC, where a GBDT did not reach DWN either.

⚠️ **The survey should be read in full for 3L-d.** It may already perform the cross-paper
normalisation this phase is attempting, in which case it is the thing to cite rather than redo.

#### Where 3L-a and 3L-c stand

**26 rows: 21 verified (read from the primary paper's own table), 5 reported (survey).** Every
number above was read from a PDF directly rather than through a summarizer — which mattered, since
the summarizer's first answer on PolyLUT was wrong.

Remaining in `pending` are all low-priority: TsetlinWiSARD, LogicNets' documented absence of an
MNIST benchmark, the 2025 PolyLUT, and reading the survey in full.

### 2026-08-13 — [literature] the "96%" rows are an upper bound, and sparsity is free accuracy

**20 verified rows.** SparseLUT Table IV re-runs the same baselines and prints them to **two
decimals** — a direct test of the rounding problem flagged this morning.

| model | its own paper | SparseLUT's re-run | Δ |
|---|---|---|---|
| NeuraLUT HDR-5L | **96%** | **95.20%** | −0.8 pp |
| PolyLUT HDR (D=2) | — | 95.42% | — |

⚠️ **NeuraLUT's own "96%" and an independent re-run of the same named model differ by 0.8 pp.**
Whether that is rounding-up or re-run variance is not resolvable from the papers. Either way the
"96%" rows should be read as an **upper bound**, which means **our margin over them is understated,
not overstated** — the safe direction, but it must be stated rather than quietly enjoyed.

Not a conflict: SparseLUT's PolyLUT rows use D=1/D=2 while PolyLUT's own MNIST HDR uses **D=4**, so
those are different configurations. Only the NeuraLUT row is a like-for-like disagreement.

#### SparseLUT itself is the interesting result, and it is free

SparseLUT changes only the **sparse connectivity**, not the LUT contents: *"SparseLUT does not alter
the number of LUT entries in the generated RTL design"*, and Fig 7 shows matching LUT/FF and no
`F_max` penalty. So it buys accuracy at **zero hardware cost**:

| | acc | LUTs |
|---|---|---|
| NeuraLUT HDR-5L | 95.20% | 54,798 |
| **SparseLUT-NeuraLUT** | **96.96%** | **54,798** (same) |
| SparseLUT-PolyLUT-Add | 97.26% | — |

**+1.76 pp for nothing.** That is directly relevant to us: our first layer's wiring is *learned*
(`learnable` mapping), which is the same lever SparseLUT pulls on architectures that use random
sparsity. It is a plausible part of why our `1x300` core lands at 630 LUTs against the paper's 692
— worth one line in the report, and **not** a claim to make without measuring.

Their "Dense" upper bounds are worth recording too: **98.55%** (PolyLUT) and **98.61%** (NeuraLUT)
for fully-connected equivalents. That brackets what this family can reach on MNIST at all, and our
98.26% at `1x2000` sits just under it.

⚠️ **Still unresolved:** the 97.5% / 75,131 PolyLUT figure is not in SparseLUT's table either (it
has no LUT column). Moved to `pending` against PolyLUT-Add.

### 2026-08-13 — [literature] ✅ the paper's own MNIST rows, and our implementation reproduces them

**14 verified rows.** DWN's Table 2 read directly — the reference the whole project is built
against, and it is the most consequential comparison in the study.

| DWN paper (xcvu9p-2, **core-only**) | acc | LUT | Fmax | latency |
|---|---|---|---|---|
| `n=6 sm` | 97.1% | **692** | 827 MHz | 2.4 ns |
| `n=6 md` | 97.9% | 1,413 | 827 MHz | 3.6 ns |
| `n=6 md` | 97.8% | 2,055 | 873 MHz | 4.6 ns |
| `n=6 lg` | **98.3%** | 4,082 | 827 MHz | 6.0 ns |

#### ⚠️ Read against the wrong convention this says we are 2.3× too big. We are not.

Our `1x300` is **1,597 LUTs** against the paper's `sm` at **692** — a 2.3× gap that would be the
headline of a careless table. **The paper's rows exclude the thermometer encoder** (brief §6,
`docs/jsc-report.md` §5.2); ours include it. Compared like for like, on core only:

| | ours | DWN paper | ratio | Δ acc |
|---|---|---|---|---|
| `1x300` vs `sm` | 96.77% / **630** | 97.1% / 692 | **0.91×** | −0.33 pp |
| `1x2000` vs `lg` | 98.26% / **4,821** | 98.3% / 4,082 | 1.18× | −0.04 pp |

**Our implementation reproduces the paper's area–accuracy trade-off.** At the small end we are
*9% smaller* for 0.33 pp less accuracy; at the large end 18% larger at the same accuracy (−0.04 pp,
well inside the 0.24 pp floor — i.e. identical).

This is the strongest available evidence that the exporter and RTL generator are correct in a way
Gate 1 cannot show: Gate 1 proves our RTL matches *our* golden model, and this shows our whole
pipeline lands where the authors' independent implementation lands.

⚠️ **The 0.33 pp gap at the small end is just above the noise floor**, so it is probably real
rather than seed scatter. Worth one line in the report, not a paragraph — candidate causes are the
`tau` schedule and the encoder's `z=3` choice, neither investigated.

#### Consequence for the report: every row must state its convention

Comparing our `dwn_top` against a core-only column **overstates our area by 1.4–2.5×** depending on
width, because the encoder share falls from 72.3% at `1x100` to 20.9% at `1x2000`. It is the
easiest mistake in this table and it is one this project has already criticised others for making.
`sweep-results.json` carries both columns; the table must use the one that matches each row.

#### Also from Table 2 — a JSC cross-check that validates the method

The same table has JSC rows: `DWN n=6 sm` at **71.1% / 20 LUTs** and **74.0% / 110 LUTs**. Our
Phase 1 measured **110 LUTs** for the 1x50 core at 73.84%. Same config, same core-only convention,
**same LUT count**. That is an independent confirmation that our reading of the convention is right
— and it is why the `1x300` comparison above can be trusted.

### 2026-08-13 — [literature] the weightless comparison lands, and it is the strongest row in the study

**10 verified rows.** BTHOWeN and ULEEN read directly from their own PDFs.

#### ✅ BTHOWeN is the closest silicon match in the entire table, and it is not close

BTHOWeN Table IV reports **real FPGA LUT counts on `xc7z020clg400-1`** — a Zynq-7000, the **same
7-series family and the same `-1` speed grade** as our XC7A35T. Every LUT-DNN row is on
`xcvu9p-…-2-i`, a Virtex UltraScale+ two process generations along. And BTHOWeN prints accuracy to
**three decimals**, so it clears the rounding problem that makes the LUT-DNN rows uncomparable.

| | acc | LUTs | cycles/inf |
|---|---|---|---|
| BTHOWeN-Small | 93.4% | 15,756 | 25 |
| BTHOWeN-Medium | 94.3% | 38,912 | 37 |
| BTHOWeN-Large | 95.2% | 151,704 | 74 |
| **this work `2x[1000,500]`** | **97.76%** | **3,464** | **5** |
| **this work `1x300`** | **96.77%** | **1,597** | **4** |

**+2.56 pp over BTHOWeN-Large at 43.8× fewer LUTs**, on comparable silicon, weightless against
weightless. `1x300` alone beats every BTHOWeN model at **9.9× fewer LUTs than the smallest**.

Also worth recording: BTHOWeN uses **thermometer encoding**, and its `Bits/Input` (2 / 3 / 6) is
exactly our `z`. So the two designs share an input stage as well as a family — this is the
like-for-like comparison `docs/mnist/phase3-ledger.md` §2.4 predicted, and it arrived better than
predicted.

#### ⚠️ ULEEN cannot go in a LUT column at all

ULEEN is the more accurate weightless model — **ULN-L reaches 98.46%**, above our best — but its
Table III is **ASIC**: area in mm² (5.22) against Bit Fusion/LeNet-5, and Table IV gives **model
size in KiB** (262). **There is no FPGA LUT number for MNIST.**

So ULEEN is comparable on **accuracy** and **model size**, and not on area. Recorded with
`lut: null` and a loud note. Putting it in a LUT column would be the single most embarrassing row
in the table, which is what §2.4 warned about — and the warning turned out to be aimed at the right
paper.

**Honest reading: ULEEN beats us on accuracy (98.46% vs 97.76%).** Our answer is area and latency,
not accuracy, and the report must say so rather than quietly comparing against BTHOWeN only.

#### The DNN baselines BTHOWeN provides are useful and must be caveated

`MLP 784-16-10` at 94.6% uses only 2,163 LUTs — *fewer than ours* — but also **8 BRAM and 28 DSP**,
and takes **846 cycles/inference** against our 4–5. It is not a LUT-only design, so a LUT-count
comparison against it is meaningless without stating the DSP/BRAM. Same for `CNN 1 (LeNet1)`:
94.7%, 5,753 LUTs, 7 BRAM, 18 DSP, **33,615 cycles**.

This is the mirror image of the encoder-convention trap: there, others exclude area we include;
here, others move area into resources we do not use at all. **Both need stating per row.**
### 2026-08-13 — [hands-on] 🏁 3M-a STOPPED at two measured points. The hands-on half is closed.

Snapshotted to `docs/results-cc-mnist/`. **Two synthesized rows plus one accuracy-only row**, and
that is the final state — the sweep is not being resumed.

| config | trees | acc% | LUTs | %dev | Fmax | cycles |
|---|---|---|---|---|---|---|
| `gbdt_d3_n5` | 50 | 84.31 | **3,653** | 17.56 | **477.3** | 3 |
| `gbdt_d4_n3` | 30 | 85.90 | 4,427 | 21.28 | **477.3** | 2 |
| `gbdt_d4_n5` | 50 | 88.34 | — | — | — | — |

`gbdt_d4_n5` is the `--no-synth` smoke test: its accuracy is real and measured, its area is not.
Kept because a third accuracy point is worth having; marked `not-synthesized` in the snapshot so
it can never be read as an area result.

#### The comparison that matters, and it is very nearly iso-area

`gbdt_d3_n5` at **3,653 LUTs** and the DWN's `2x[1000,500]` at **3,464 LUTs** are within 5% of the
same area, on the same part, at the same clock target, under the same encoder-inclusive
convention:

| | acc% | LUTs | %dev | Fmax |
|---|---|---|---|---|
| **DWN `2x[1000,500]`** | **97.76** | 3,464 | 16.65 | 103.8 |
| conifer `gbdt_d3_n5` | 84.31 | 3,653 | 17.56 | **477.3** |
| | **−13.45 pp** | +5% area | | **4.6×** |

**13.45 pp at matched area.** That is the number this half of the phase exists to produce, and two
rows were enough to produce it — an iso-area comparison needs a matched pair, not a curve.

⚠️ **And conifer wins decisively on speed: 477.3 MHz against 103.8, at 3 cycles against 5.** It
repeated across both configs, so it is a property of the flow rather than a fluke, and it belongs
in the table with the same weight as the area result. A GBDT's critical path is one comparator
tree; the DWN's is an encoder plus LUT layers plus a popcount.

#### Why it stopped here — three reasons, and only the first is new

1. **Background runs kept being killed externally**, twice within an hour, both mid-HLS. Config 3
   (`gbdt_d3_n10`) lost ~58 minutes of HLS and produced no `csynth.xml`, so nothing was
   salvageable. The cost of another point was not "one hour" but "one hour, repeatedly, until it
   happens to land between kills".
2. **HLS is disproportionately expensive here.** 350 comparators took 31 min, 450 took 42, 700 was
   unfinished at 58 — driven by ensemble size *and* the 784-wide input interface.
3. **JSC's Phase 3 already carries the trend argument** with a full 14-point curve on the same
   mechanism. MNIST does not need to re-demonstrate that more trees do not close the gap.

⚠️ **What is lost, stated plainly:** MNIST has **no conifer curve and no measured ceiling**. The
supported sentence is *"the largest conifer ensemble built was 4,427 LUTs at 85.90%"* — never
*"conifer's ceiling on this part is X"*. Same form of limitation Phase 2 recorded for the DWN
ladder stopping at 33% of device.

#### Correction to my own recommendation

**I over-recommended running this sweep.** The reasoning that the hands-on half was needed to
anchor cross-silicon literature rows was sound, but I sized it as a 13-point curve when an
iso-area *pair* was sufficient, and it took two rounds of the question being asked before I
updated. MNIST Phase 3's genuinely new content is the literature half — ULEEN and BTHOWeN
benchmark MNIST and report no JSC, so it is DWN against its own family for the first time — and
that needs no Vivado at all.

#### A mislabel in `snapshot()`, fixed, affecting JSC's console output too

The summary printed non-`ok` rows as "over-device". But `status='ok'` means **synthesis
succeeded**, not that the design fits — so the split was really "synthesized / didn't". JSC's
snapshot has been reporting *"10 fit, 4 over-device"* when the truth is **"10 fit, 0 over-device,
4 synth-failed"**: those four are the largest configs and they failed synthesis outright rather
than reporting a LUT count over the part.

✅ **No published claim is affected** — "over-device" appears nowhere in `docs/phase3-report.md`
or `docs/jsc-report.md`. The committed JSON and CSV are byte-identical; only the printed line changed.

### 2026-08-13 — [hands-on] ❌ 3M-b hls4ml is CUT, and 3M-a is trimmed to six points

**Both predictions from 3M-0 are confirmed on the first two configs**, and the scope decisions
below follow from that plus a measured cost problem, not from impatience.

| config | trees | acc% | LUTs | %dev | Fmax | LUTs/tree |
|---|---|---|---|---|---|---|
| `gbdt_d3_n5` | 50 | 84.31 | 3,653 | 17.56 | **477.3** | 73.1 |
| `gbdt_d4_n3` | 30 | 85.90 | 4,427 | 21.28 | **477.3** | 147.6 |

**Prediction 2 (per-tree cost) — confirmed precisely.** A depth-*d* tree is 2^*d*−1 comparators,
so 73.1 at depth 3 predicts 156 at depth 4; measured **147.6**. Tree area is set by depth and
comparator width, **not** by how many features exist to choose from — 784 features cost conifer
nothing in area over JSC's 16.

**Prediction 1 (accuracy) — confirmed, by a wide margin.** conifer spends **4,427 LUTs for
85.90%**; the DWN's `1x100` gets **92.98% in 845**. Five times the area, seven points behind. On
JSC the gap was arguable; here it is not close.

⚠️ **One genuine conifer advantage, and it repeated across both configs: 477.3 MHz against our
103.8**, at 3 cycles. That is a real result and belongs in the table with the same weight as the
area result, not as a footnote.

#### ❌ 3M-b (hls4ml on MNIST) is not being run. Reasons, so this is a decision and not a gap.

1. **A published MNIST hls4ml row already exists** — ternary (Duarte), 260,092 LUTs, in the 3L-a
   table. The axis is covered by citation in a way conifer's is not: **there is no published
   MNIST GBDT-on-FPGA row at all**, which is exactly why 3M-a is the half worth keeping.
2. **It will not fit.** On JSC, hls4ml fit only at quarter width. MNIST's first dense layer scales
   with input count — **784 against 16** — so the measurement would be "does not fit", which the
   260,092-LUT literature row already implies on a part 57× larger than ours.
3. **JSC Phase 3 measured the hls4ml axis in full, under control.** The method is demonstrated;
   repeating it on a dataset where the answer is known adds no evidence.

⚠️ **What this costs, stated plainly:** MNIST has no *controlled* quantized-MLP comparison — only
a cited one on different silicon. Any sentence comparing us to hls4ml on MNIST must say so.

#### The trim: 13 → 9 → 6 configs, the second one for time rather than area

The first trim dropped four points predicted several times over the device. The second is about
**HLS cost, which scales with total comparators AND with the 784-wide input interface**:

| comparators | HLS time |
|---|---|
| 350 (`d3_n5`) | 31 min |
| 450 (`d4_n3`) | 42 min |
| 700 (`d3_n10`) | still running at 50 min when stopped |

Extrapolating, the four largest survivors (1,400–1,890 comparators) were **2–3 hours each** —
10–13 hours for the tail alone. Six points remain, five below 47% of device, plus `(5, 5)` at
77.8% kept as the single high-accuracy anchor: the open question is whether *more trees* ever
close the gap, and only the largest surviving ensemble can answer it. Cheapest-first ordering
runs it last, so it is abandonable without losing the other five.

⚠️ **This loses the measured frontier edge.** `(6, 3)` at ~95% of device was the one point whose
answer was genuinely unknown. Recorded as a stated limit, the same way Phase 2 recorded the DWN
ladder stopping at 33% — *"the largest conifer config built was X"*, never *"the largest that
fits is X"*.

#### ⚠️ Vitis HLS stalled once, and it is silent when it does

`gbdt_d4_n3`'s first attempt wedged in a `clang-tidy` pass: processes alive, **zero CPU
system-wide for 45 s, zero new files**, no error in any log. Killed and restarted; it completed
normally on the second attempt, so it is not a property of the design.

**Liveness cannot be judged from `vitis_hls`'s own CPU** — the work happens in child processes
(`clang`, `clang-tidy`), so the parent reads zero while the flow is perfectly healthy. The test
that distinguishes them is a system-wide CPU sample plus a file count over the same window.

### 2026-08-13 — [hands-on] 3M-0: `cc/` is dataset-aware, and two predictions recorded before measuring

`cc/conifer/run_conifer.py` reads `datasets/` — same option-1 treatment `dse/` got in Phase 2,
rather than a parallel MNIST path. `--dataset mnist` selects the OpenML name, feature and class
counts, scaler, split, sweep grid and every output path.

**JSC is provably unchanged**, which is the gate:

```
sweep identical to the original : True (14 configs, same order)
paths                           : build/cc/conifer, docs/results-cc   (unchanged)
X_train sha256[:16]             : 9815d799940ee527
```

That hash is the real check — `build/cc/jsc_data.npz` was written *before* the refactor and
`load_data()` still returns it bit-identically.

#### ⚠️ The GBDT grid does NOT transfer between datasets

xgboost builds **`n_estimators × classes`** trees for multiclass. At MNIST's ten classes the same
`n_estimators` therefore costs **twice the trees, and roughly twice the area**, that it does at
JSC's five. Copying JSC's grid would have compared a 100-tree ensemble against a 50-tree one and
recorded them as the same point.

MNIST's grid halves `n_estimators` so total tree count brackets the same region — JSC's `(4, 20)`
is 100 trees and MNIST's `(4, 10)` is also 100. Both grids live in `datasets/` as data, with the
reasoning beside them.

#### ⚠️ `SNAPSHOT_DIR` was assigned twice, 380 lines apart

Once near the top and again at `docs/results-cc` below `save()`. The second wins, so every
`--dataset mnist --snapshot` would have **overwritten JSC's committed rows** while printing a
plausible message. Removed, with a comment where it used to be saying why it is not there.

Found by reading rather than by running, which is the only way this one surfaces: the failure is
silent, and its symptom is a corrupted artifact in a *different* dataset's directory.

#### Predictions, recorded before the first measurement

Both are falsifiable, and a surprise in either is a genuine finding rather than a bug.

| | prediction | reasoning |
|---|---|---|
| **conifer accuracy** | **will not reach the DWN frontier**, and by a wider margin than on JSC | A GBDT splits one feature at a time. JSC's 16 features are engineered physics variables, each individually informative; MNIST's 784 raw pixels are not — no single pixel carries much, and trees cannot form the linear combinations that make pixels useful |
| **conifer area** | **per-tree LUT cost close to JSC's ~160 at depth 4** | Tree area is set by depth (a depth-*d* tree is 2^*d*−1 comparators) and comparator width, not by how many features exist to choose from. If this is wrong, the mechanism is worth understanding |

⚠️ The second prediction has a consequence worth stating in advance: if per-tree cost holds and
MNIST needs *more* trees for accuracy, MNIST's conifer curve sits strictly worse than JSC's on
both axes — which would make the comparison against DWN more lopsided here, not less.

### 2026-08-13 — [literature] 3L-a started: four rows verified, and three problems found

`cc/literature/mnist_literature.json` created, schema copied from `jsc_literature.json` so
`cc/literature/table.py` and `plot.py` can read both.

**NeuraLUT Table III, read directly from the PDF** (not via a summarizer — the JSC phase got burned
twice that way):

| method | model | acc | LUT | FF | BRAM | Fmax | latency |
|---|---|---|---|---|---|---|---|
| NeuraLUT | HDR-5L | 96% | 54,798 | 3,757 | 0 | 431 MHz | 12 ns |
| PolyLUT | HDR | 96% | 70,673 | 4,681 | 0 | 378 MHz | 16 ns |
| FINN | SFC-max | 96% | 91,131 | — | 5 | 200 MHz | 310 ns |
| hls4ml | ternary (Duarte) | 95% | 260,092 | 165,513 | 0 | 200 MHz | 190 ns |

✅ **This confirms the figures already quoted in `docs/mnist/phase1-report.md`** (PolyLUT 96% /
70,673, NeuraLUT 96% / 54,798) — they are exactly what the paper prints.

#### ⚠️ Problem 1: the sources round accuracy to whole percent, and that is coarser than our noise floor

NeuraLUT's Table III prints `96%` and `95%`. A row printed as "96%" spans **±0.5 pp — twice our
0.24 pp noise floor.** Two such rows cannot be ranked against each other at all, and ours can only
be compared to them when the intervals are disjoint.

For the headline that does hold: **ours 97.76% ± 0.24 → [97.52, 98.00]** against **96% ± 0.5 →
[95.5, 96.5]**. Disjoint, so that gap is real. Any comparison inside ~1 pp is not.

**Added `accuracy_precision` to the schema** (`whole_pct` / `one_dp` / `two_dp`) so a table can
refuse to rank rows it cannot support. This did not come up on JSC because those sources print two
decimals.

#### ⚠️ Problem 2: two different PolyLUT MNIST numbers are in circulation

**96% / 70,673 LUTs** (NeuraLUT Table III, verified) versus **97.5% / 75,131 LUTs** (secondary
source, unverified). Different on both axes, both labelled PolyLUT "HDR".

This is the same *shape* as JSC Phase 3's two-datasets problem — a number everyone quotes that
turns out to mean two things. **Resolve before tabulating PolyLUT**: read `arXiv:2309.02334`
directly and record which configuration each corresponds to. First item in `pending`.

#### ⚠️ Problem 3: all four rows are on `xcvu9p`, ~1.18M LUTs

NeuraLUT's MNIST model at 54,798 LUTs is **2.6× our entire XC7A35T**. So LUT *count* is comparable
as a number, but "fits on the board" is not, and the speed grades differ — `xcvu9p-…-2-i` against
our `-1`. The JSC plan's §4.2 silicon caveat applies unchanged.

#### Where that leaves us, on the numbers verified so far

Our best config that meets the **board** clock — `2x[1000,500]`, 97.76%, 3,464 LUTs, 103.8 MHz:

| vs | their LUTs | ratio |
|---|---|---|
| NeuraLUT HDR-5L | 54,798 | **15.8×** |
| PolyLUT HDR | 70,673 | **20.4×** |
| hls4ml ternary | 260,092 | **75.1×** |

⚠️ Provisional: 4 rows of a table that should have ~15, and the weightless family (ULEEN,
BTHOWeN) — the comparison that actually matters for DWN — is not in it yet.

#### Next

`pending` in the JSON is ordered. PolyLUT's discrepancy first, then ULEEN/BTHOWeN, then sweep the
papers already read for JSC for MNIST rows — that is the cheapest source of verified data since
those PDFs are already vetted.
