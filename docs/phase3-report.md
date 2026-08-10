# Phase 3 — the controlled comparison: what we measured, what we could not, and two ways this literature lies

**Question (brief §10):** for the same task on the same silicon, how does hand-written weightless
RTL compare to the standard FPGA-ML toolchains — and where does it sit against the published
LUT-DNN literature?

Two halves, run on two machines. The *hands-on* half (conifer **and hls4ml**) was measured through the same
`scripts/build.tcl` flow as all 54 Phase 2 DWN configs. The *literature* half is citation and
plotting only. Running log with the dated detail: `docs/phase3-ledger.md`.

⚠️ **The most useful thing in this report is not a number.** It is that the standard JSC
comparison table — the one reproduced in the DWN paper, in Mecik & Kumm, and in the 2025 survey —
**mixes two different datasets and two different area conventions.** §3 is that finding. Every
result below is stated in a way that survives it.

---

## 1. Headline results

| | |
|---|---|
| **DWN vs conifer, identical silicon** | **+1.5 to +1.7 pp accuracy at every area budget**; 2.3–6.0× fewer LUTs at matched accuracy |
| **Can a GBDT match DWN here?** | **No.** conifer tops out at **74.88%** using 73.9% of the device. DWN reaches that at **3,381 LUTs** and continues to 76.35% |
| **Where conifer wins** | **Speed, decisively** — 477.3 MHz vs our 101–147 MHz; 4.2–16.8 ns vs our 27–40 ns. In *cycles* they are comparable (2–8 vs our 4) |
| **DSP/BRAM** | 0/0 for **all** 14 conifer and **all** 52 DWN configs. The fitting hls4ml design needs **53 DSPs — 59% of this part's 90** |
| **hls4ml on our part** | Fits only at **quarter width + 12-bit + 4× reuse**: 75.67% at 8,749 LUTs, 53 DSP, **34 cycles / II=4**. The published architecture is 2.2× over even at 16× reuse |
| **hls4ml vs DWN, matched area** | DWN is smaller, **+0.38 pp**, 0 DSP, **8.5× lower latency, 4× the throughput** |
| **Best published LUT count at ≥76% on our dataset** | NeuraLUT-Assemble, **1,780 LUTs** — against our 12,751 |

---

## 2. What was compared, and on what

Everything in the "measured" column below went through `scripts/build.tcl`, out-of-context,
`xc7a35tcpg236-1`, 10 ns, same strategy, same thread count.

| source | rows | status |
|---|---|---|
| DWN (this project) | **51 fitting of 54** — 42 of those also close 100 MHz | measured, Phase 2 |
| conifer (GBDT) | 10 fitting of 14 | measured, Phase 3 |
| hls4ml (quantized MLP) | 1 fitting of 6 | measured, Phase 3 |
| published LUT-DNN family | 32 rows, 11 methods | cited, `cc/literature/jsc_literature.json` |

The three DWN counts are all used somewhere and are easy to conflate, so: **54** records in
`docs/results/sweep-results.csv`, **52** built (2 are `gate1-failed` — the linear-encoding configs
of Phase 2 §5.3), **51** fit the device (`1x2000` builds but measures 102.80%), and **42** both fit
and meet the board's 100 MHz. §1's "all 52 DWN configs" for DSP/BRAM is the *built* count, since
`1x2000` was measured at 0/0 too.

Figures: `docs/results-cc/jsc-openml.png`, `docs/results-cc/jsc-cernbox.png`.
Regenerate with `.venv\Scripts\python.exe cc\literature\plot.py --snapshot`.

---

## 3. ⚠️ Two ways this literature lies, and both had to be fixed before any table meant anything

### 3.1 "JSC" is two datasets, ~1.05 pp apart

| | JSC-**OpenML** | JSC-**CERNBox** |
|---|---|---|
| source | `hls4ml_lhc_jets_hlf`, OpenML 42468 | CERNBox LHC Jets |
| instances | ~830,000 | 986,806 |
| **who uses it** | **DWN, TreeLUT, hls4ml — and us** | LogicNets, PolyLUT, PolyLUT-Add, NeuraLUT, AmigoLUT, ReducedLUT, SparseLUT |

NeuraLUT-Assemble (arXiv:2504.00592 §5) states it plainly: *"Both datasets target the same jet
classification task… Experimentally, we observed that models trained on the OpenML dataset achieve
higher accuracy."*

**The offset is measured within-method, twice**, in FPGN's Table V — the only clean way to get it:
NeuraLUT-Assemble scores 76.0% (OpenML) vs 75.0% (CERNBox); FPGN scores 76.0% vs 74.9%. That is
**~1.05 pp, seven times our 0.15 pp noise floor**, and larger than nearly every difference our
own Pareto frontier argues about.

**We are on OpenML** — `third_party/DWN`'s tutorial calls `openml.datasets.get_dataset(42468)`,
our training notebooks call `fetch_openml('hls4ml_lhc_jets_hlf')`, and our 166k test set is 20% of
~830k.

**Consequence:** the comparison everyone quotes — DWN against PolyLUT/NeuraLUT/LogicNets — is
cross-dataset. Those are CERNBox rows. Our real peer group on OpenML is much smaller and much
tougher: DWN, TreeLUT, NeuraLUT-Assemble, FPGN, hls4ml.

`cc/literature/plot.py` refuses to draw both datasets on one axis, and `table.py` refuses to print
them in one table. This is enforced in code rather than remembered.

### 3.2 LUT counts do not all include the input encoder

The DWN paper reports `lg` at **4,972 LUTs — core only, no thermometer encoder.** With the encoder
it is **7,011** (Mecik & Kumm's `DWN-PEN+FT`). Ours are always totals. Quoting one against the
other is wrong in both directions, and this is exactly why brief §6 requires core and encoder
reported separately.

Three different published numbers exist for DWN `lg`: **4,972** (core only), **7,011** (with
encoder), **6,302** (FPGN's re-implementation, convention unstated). Any comparison must say which.

**For the non-DWN rows the convention is `n/a`, and this was verified, not assumed.** TreeLUT
quantises inputs *"as a pre-processing step"* — host-side, exactly as our StandardScaler is. And
Mecik & Kumm settle it for the family: *"In previous performance evaluations, only the resource
usage of the LUT layer and the classification logic was reported"* — said **of DWN**. DWN was the
outlier; the LUT-based architectures report complete designs. They feed quantised inputs straight
into LUT address lines and have no expansion stage. DWN needs an encoder because thermometer
coding turns 16 features into 3,200 bits.

### 3.3 Both defects originate upstream

DWN v5's own JSC table lists `hls4ml` (OpenML) beside `PolyLUT` and `NeuraLUT` (CERNBox), and
reports its own LUTs core-only against their full designs. **Both mistakes are in the primary
source and propagate by copying.** Mecik & Kumm's Table II and the 2025 survey inherit the dataset
mixing; the survey splits JSC only by accuracy band and never mentions two data sources exist.

---

## 4. DWN vs conifer — the same-silicon result

Both through our flow, same part, same clock, same strategy.

**Iso-area — best accuracy within a LUT budget:**

| budget | DWN | conifer | gap |
|---|---|---|---|
| 4,000 | `1x360 z=25` 75.27% | `gbdt_d3_n10` 73.64% | **+1.63 pp** |
| 8,000 | `1x800 z=50` 75.95% | `gbdt_d5_n5` 74.36% | **+1.59 pp** |
| 12,751 | `1x2400 z=50` 76.18% | `gbdt_d6_n3` 74.50% | **+1.68 pp** |
| 20,800 | `1x1600` 76.35% | `gbdt_d3_n40` 74.88% | **+1.48 pp** |

**Iso-accuracy — smallest design reaching a target:**

| target | DWN | conifer | ratio |
|---|---|---|---|
| 73.6% | 1,619 | 3,774 | **2.3×** |
| 74.2% | 2,541 | 7,602 | **3.0×** |
| 74.5% | 2,541 | 15,363 | **6.0×** |
| ≥74.9% | 3,381 | **never reaches it** | — |

**A GBDT does not reach DWN's accuracy on this part at any size that fits.**

### 4.1 Where conifer wins, and it is not close

**Speed.** conifer closes at **477.3 MHz** on the same part where our headline design manages
**101.3 MHz**, and its latency is **4.2–16.8 ns** against our **27–40 ns**.

The reason is structural: our critical path is a 2,400-wide popcount and argmax; a boosted tree is
a shallow comparison cascade. Reported in **cycles** the two are comparable — conifer 2–8, ours 4
— which is precisely why brief §6 requires cycles alongside nanoseconds.

**The fair one-line summary: DWN wins accuracy-per-LUT; conifer wins speed.** Anyone quoting only
the first half is quoting us selectively.

### 4.2 What conifer cost to get right

Four silent failures, all recorded in the ledger. The one worth repeating: **xgboost ≥ 2.0
auto-fits a per-class base score that conifer 1.9 cannot read**, emitting
`init_predict = [-4.965, NaN, -4.965, -5.742, -4.862]`. A single NaN makes that class's score NaN
for every sample and sends the argmax arbitrary — **127,034 of 166,000 predictions wrong**, while
producing entirely plausible-looking HDL. Caught by an independent numpy evaluator of conifer's
own emitted ensemble JSON: the same golden-model pattern Gate 1 uses.

---

## 5. hls4ml, measured on our part — it fits only at quarter width

**Run after all.** Every predicted toolchain blocker turned out not to exist: hls4ml 1.3.0 needs
no WSL, no second Vivado install, no `vitis_hls` shim and no TensorFlow. Its Vitis backend already
issues `vitis-run --tcl build_prj.tcl --mode hls`, and `convert_from_pytorch_model` uses the torch
already pinned.

The published architecture is **16→64→32→32→5** (Duarte/Fahim), trained in PyTorch on our split
and pushed through the same `build.tcl` as everything else.

| config | reuse | precision | acc (float) | LUTs needed | vs 20,800 | DSP |
|---|---|---|---|---|---|---|
| `64/32/32` **as published** | 1 | `<16,6>` | 76.69% | **259,492** | 12.5× over | 3,214 |
| `64/32/32` | 4 | `<16,6>` | 76.69% | 189,608 | 9.1× over | 1,064 |
| `32/16/16` | 4 | `<16,6>` | 76.33% | 52,927 | 2.5× over | 340 |
| `64/32/32` | 16 | `<16,6>` | 76.69% | 45,844 | 2.2× over | 266 |
| `32/16/16` | 4 | `<12,6>` | 76.33% | 26,883 | 1.3× over | 340 |
| **`16/8/8`** | 4 | `<12,6>` | **75.67%** | **8,749** | ✅ **42.1% — FITS** | **53** |

**The published architecture does not fit at any reuse factor tested** — still 2.2× over at 16×
time-multiplexing, at full accuracy. Fitting requires quarter-width layers *and* 12-bit precision
*and* 4× reuse, and costs **1.02 pp**.

**Reuse beats width as an area lever.** 1→4 bought only 1.4×, 4→16 bought 4.1×; and `64/32/32` at
reuse 16 is *smaller* than `32/16/16` at reuse 4 while scoring 0.36 pp better. Reuse costs latency,
not accuracy — which is why it is the right knob, and why the fitting design pays for it below.

### 5.1 At matched area, DWN wins every column

| | LUTs | DSP | BRAM | latency | throughput | accuracy |
|---|---|---|---|---|---|---|
| hls4ml `16/8/8` | 8,749 | **53** | 2 | **34 cyc** | II=4 | 75.67% † |
| **DWN `1x1200 z=50`** | **8,444** | **0** | **0** | **4 cyc** | **II=1** | **76.05%** |

Smaller, more accurate, no DSPs, no BRAM, **8.5× lower latency and 4× the throughput**. DWN's
headline `1x2400 z=50` reaches **76.18% at 12,751 LUTs** — an accuracy hls4ml never reaches at any
size that fits this part.

**The DSP claim is now measured rather than cited.** 53 DSPs is **59% of this part's 90**, against
**0** for all 52 DWN configs and all 14 conifer configs. This was the gap an earlier draft of this
report flagged as unexercised on our own silicon.

† ⚠️ **The hls4ml accuracy is the FLOAT model's — an upper bound.** Measuring the real
`ap_fixed<12,6>` figure needs `hls_model.predict()`, which compiles the generated C++, and no C++
compiler is available here. conifer avoided this because its model is a declarative ensemble JSON
we could evaluate in numpy; hls4ml's is a C++ dataflow program with no equivalent artifact. **The
true number is lower, so every comparison above already favours hls4ml.**

### 5.2 Two silent failures, one of which nearly shipped

**Weight ROMs read as zero.** At `reuse>1` hls4ml moves weights into ROMs loaded by
`$readmemh("./x.dat", rom0)` — a *relative* path. Staging the Verilog elsewhere left every ROM
unresolved, the weights read as zero, the synthesizer folded the dead arithmetic away, and a
64/32/32 MLP reported **235 LUTs, 0 DSP, status `ok`**. Two rows were recorded that way before the
numbers were disbelieved. Staging now carries the `.dat` files and rewrites the paths, and a guard
refuses to synthesize RTL with undefined modules or missing ROMs. conifer was verified unaffected
(inline constants, no `$readmemh`).

**Latency read zero, because of a unit.** Latency-strategy reports are in ns; Resource-strategy
reports switch to **us** and report Pipeline Type `dataflow` rather than `yes`. A regex requiring
`' ns|'` silently returned nothing, hiding the II=4 result entirely — and a missing `+ Detail:`
boundary made the `reuse=1` row read 4 cycles when the true figure is 30. All 14 conifer rows were
re-verified against the corrected parser and are unchanged.

---

## 6. Where we sit against the published field, on our dataset

JSC-OpenML only. DWN rows shown in both conventions.

⚠️ `PENDING-AUTHOR-REPLY` — the DWN rows assume the paper's numbers are **JSC-OpenML**. See §8;
`grep -rn PENDING-AUTHOR-REPLY docs/ cc/` finds every claim that moves if that is wrong.

| Method | Acc | LUT | Encoder | Part |
|---|---|---|---|---|
| **this project** `1x1600` | **76.35%** | 18,777 | incl | `xc7a35t-1` |
| DWN `lg-2400` | 76.3% | 4,972 | core only | `xcvu9p-2` |
| DWN `lg-2400` PEN+FT | 76.3% | 7,011 | **incl** | `xcvu9p-2` |
| **this project** `1x2400 z=50` | **76.18%** | **12,751** | incl | `xc7a35t-1` |
| hls4ml (Fahim et al.) | 76.0% | 63,251 | n/a | `xcvu9p` |
| TreeLUT (I) | 76.0% | 2,234 | n/a | `xcvu9p-2` |
| **NeuraLUT-Assemble** | 76.0% | **1,780** | n/a | `xcvu9p-2` |
| FPGN | 76.0% | 3,345 | ? | `vu9p` |
| conifer `gbdt_d3_n40` | 74.88% | 15,363 | n/a | `xc7a35t-1` |

**Read this honestly.** The best published LUT count at ≥76% on our dataset is **1,780**, against
our 12,751. We are not competitive on raw area with the specialised LUT-DNN compilers.

What is different about our numbers, and must be said whenever they are quoted:

- **Encoder included.** Ours are complete designs. The 4,972 figure is not.
- **A `xc7a35t-1` at 100 MHz**, not a `xcvu9p-2` at 700 MHz — a ~$150 board against a data-centre
  part. LUT counts roughly transfer; Fmax and nanoseconds do not.
- **The comparable row is DWN `lg` PEN+FT at 7,011**, the only one sharing both our dataset and
  our convention.

---

## 7. A finding from the literature half: our encoder was 5.9× too large

Mecik & Kumm's encoder costs **201 LUTs** where ours costs **1,519** on the same 50-node model at
the same `z=200`. Chasing that produced the largest single area result of the project.

**The cause is comparator width, entirely.** They normalise features to [−1,1) and quantise to
6–9 bits; we carried a global Q3.12 16-bit word whose 3 integer bits exist only because our
features are standard-scaled to roughly ±4.5. At 8 bits ours is **182 LUTs** against their 201 —
no clever structure was missing, only narrower numbers.

Measured on `1x2400 z=50`, accuracy on all 166k, area out-of-context:

| word | LUTs | vs today | accuracy-safe? |
|---|---|---|---|
| 16 | 5,753 | 1.00× | — |
| 12 | 4,157 | 1.38× | yes |
| **11** | **992** | **5.80×** | **yes (−0.142 pp)** |
| 10 | 891 | 6.46× | no (−0.219 pp) |

**There is a cliff between 12 and 11 bits** — 4.7× for two bits — and the accuracy limit lands on
the cheap side of it by exactly one bit. That is luck, not design: one bit lower and the usable
saving would have been 1.38×.

**It does not change Phase 2's conclusions**, and the reason matters: every config that beat our
headline failed on **timing**, not area, and the encoder is not on the critical path (347 MHz vs
the core's 101.2). Shrinking it ~6× does not move the wall — which is the strongest confirmation
of *"the wall is timing, not area"* we have.

⚠️ **Not adopted.** Projected 12,751 → ~7,990 LUTs (~38% of device), but it buys area on a design
whose binding constraint is timing, and adopting it costs a renorm-capable emitter, host-side
scaler changes and Gate 1 regeneration. Recorded as a finding; `docs/phase3-ledger.md` carries the
full numbers.

---

## 8. Limitations

- **hls4ml's accuracy is the float model's, not the quantized design's** (§5). `hls_model.predict()`
  compiles C++ and no compiler is available here, so the real `ap_fixed<12,6>` figure is lower than
  75.67% — every hls4ml comparison in this report therefore favours hls4ml.
- **⚠️ `PENDING-AUTHOR-REPLY` — which JSC dataset the DWN paper used is unresolved.** The paper says *"as in the NeuraLUT
  paper"* (CERNBox); its released code, FPGN's classification, our own reproduction within 0.12 pp,
  and its accuracy level all say OpenML. Four lines of evidence against one sentence. **Our
  headline comparison against DWN `lg` is only valid if OpenML.** The authors have been asked; no
  reply yet.
- **LLNN's dataset is unknown** (IEEE TCAS-I, paywalled). Two rows are on neither figure.
- **SparseLUT, BitLogic, KANELE not extracted** — known to exist, numbers not pulled.
- **Two baselines, not "the standard toolchains".** conifer and hls4ml are both measured here, but
  they are two tools; the wider LUT-DNN family is cited, not re-synthesized (§6).
- **Datapath precision was fixed** at Q3.12 across all 54 Phase 2 configs, and §7 shows it was the
  largest unswept axis in the project.

---

## 9. Reproducing this

```
.venv\Scripts\python.exe cc\literature\table.py                  # combined table, OpenML
.venv\Scripts\python.exe cc\literature\table.py --dataset cernbox
.venv\Scripts\python.exe cc\literature\plot.py --snapshot        # both figures
.venv\Scripts\python.exe cc\conifer\run_conifer.py --sweep       # needs Vivado + HLS
```

Committed evidence: `docs/results-cc/` (conifer measurements + both figures),
`cc/literature/jsc_literature.json` (32 published rows, each with source and confidence).

---

## 10. Pointers

- `docs/phase3-ledger.md` — the dated log, including every correction and retraction
- `docs/phase3-plan.md` — what Phase 3 set out to do
- `docs/phase2-report.md` — the DWN frontier this compares against
- `cc/literature/jsc_literature.json` — per-row source, dataset, convention, confidence
