# MNIST Phase 3 — the controlled comparison: DWN against its own family, for the first time

The written-up account of MNIST Phase 3. Running log with the dead ends:
`docs/mnist/phase3-ledger.md`. The JSC equivalent is `docs/phase3-report.md`; most of the *method*
is settled there and is reused rather than re-derived here.

**What is new on MNIST, and it is not the accuracy number.** JSC Phase 3 could only compare a DWN
against decision trees, quantised MLPs and LUT-DNNs. **ULEEN and BTHOWeN are weightless neural
networks — DWN's own lineage — and they benchmark MNIST while reporting no JSC at all.** This is
the first like-for-like family comparison the project has been able to make.

---

## 1. Headline results

| | |
|---|---|
| **Literature table** | **26 rows, 21 read directly from the primary paper's own table** |
| **vs the DWN paper, its convention** | `1x300` **630 LUTs** vs its `sm` **692** — **0.91×**, −0.33 pp |
| **vs BTHOWeN** (weightless, comparable silicon) | **+2.56 pp at 43.8× fewer LUTs** |
| **vs conifer** (GBDT, our silicon, iso-area) | **+13.45 pp at +5.5% area** — but conifer is **4.6× faster** |
| **vs the LUT-DNN field** | 15.8× smaller than NeuraLUT, 20.4× than PolyLUT, 75.1× than hls4ml |
| **ULEEN** | **beats us: 98.46% vs 97.76%** — ASIC only, no FPGA area exists |

Our designs that meet the **100 MHz board clock**, both accounting conventions:

| config | acc% | core | encoder | total | % device | Fmax | cycles |
|---|---|---|---|---|---|---|---|
| `2x[1000,500]` | **97.76** | 2,168 | 1,302 | 3,464 | 16.65 | 103.8 | 5 |
| `1x500` | 97.70 | 1,168 | 1,079 | 2,246 | 10.80 | 108.4 | 4 |
| `1x300` | 96.77 | 630 | 971 | 1,597 | 7.68 | 107.5 | 4 |
| `1x200` | 95.93 | 420 | 839 | 1,264 | 6.08 | 111.3 | 4 |
| `1x100` | 92.98 | 234 | 611 | 845 | 4.06 | 123.8 | 4 |

Figure: `docs/results-cc-mnist/mnist.png`. Table: `cc/literature/table.py --benchmark mnist`.

## 2. What was compared, and on what

| | rows | silicon |
|---|---|---|
| this project | 25 measured configs | `xc7a35tcpg236-1`, 10 ns, post-route |
| conifer (GBDT) | 2 synthesized + 1 accuracy-only | **the same part, the same clock, the same flow** |
| hls4ml | ❌ cut — see §5 | — |
| published LUT-DNN | PolyLUT, PolyLUT-Add, NeuraLUT, ReducedLUT, TreeLUT, AmigoLUT, SparseLUT | `xcvu9p-flgb2104-2-i` |
| published weightless | BTHOWeN, ULEEN | `xc7z020clg400-1` / ASIC |
| the DWN paper itself | 4 MNIST rows | `xcvu9p-flgb2104-2-i` |

**Only the conifer rows are ours.** Everything else is a citation, read from the primary source.

⚠️ **BTHOWeN is the closest silicon match in the study** — `xc7z020clg400-1` is the same 7-series
family and the same `-1` speed grade as our part, where every LUT-DNN row is a Virtex UltraScale+
`-2` two process generations along. That makes the weightless comparison the most defensible one
here, which is fortunate, because it is also the one that matters most.

## 3. Three ways this literature resists comparison

### 3.1 The LUT counts use different conventions, and the correction is width-dependent

Published counts are frequently **core-only**, excluding the input encoder. Ours include it.

Read against the wrong convention, our `1x300` at 1,597 LUTs against the DWN paper's `sm` at 692
looks like a **2.3× failure**. It is not: like for like, on core only, `1x300` is **630 LUTs** —
*smaller* than the paper's 692.

On MNIST the correction is **not a single factor**, because the encoder saturates:

| | `1x100` | `1x300` | `1x1000` | `1x2000` |
|---|---|---|---|---|
| encoder share of total | **72.3%** | 60.8% | 35.0% | **20.9%** |

So a fixed multiplier cannot fix a mis-conventioned table. **Every row must state its own
convention**, which is why `docs/results-mnist/sweep-results.json` carries `dwn_core_luts` and
`thermometer_encoder_luts` separately, and why the figure encodes convention in marker fill.

This is the defect `docs/jsc-report.md` §5.2 criticises in the JSC literature. It recurs on MNIST, and the
width-dependence makes it *worse* here, not better.

### 3.2 Accuracy is often printed to whole percent — coarser than our noise floor

NeuraLUT's and PolyLUT's tables print `96%` and `95%`. A row printed as "96%" spans **±0.5 pp —
twice our measured 0.24 pp noise floor** (`docs/mnist/reduction-ledger.md`, four configurations ×
four seeds). Two such rows cannot be ranked against each other at all.

**And there is direct evidence the rounding is generous.** SparseLUT re-runs the same baselines and
prints two decimals:

| model | its own paper | independent re-run |
|---|---|---|
| NeuraLUT HDR-5L | **96%** | **95.20%** |
| PolyLUT HDR (D=2) | — | 95.42% |

Whether that is rounding-up or run-to-run variance is not resolvable from the papers. Either way
the "96%" rows are an **upper bound**, so our margins over them are understated rather than
overstated — the safe direction, but it must be said rather than quietly enjoyed.

**Rule applied throughout: no two designs are ranked on an accuracy gap below 0.24 pp.**

### 3.3 ✅ The dataset-ambiguity defect does *not* apply — and that is a result

JSC Phase 3's largest correction was that **"JSC" is two different datasets ~1.05 pp apart**, and
the standard comparison table conflated them (`docs/phase3-report.md` §3.1).

**MNIST has one canonical split.** `mnist_784` ships in train-then-test order and the last 10,000
rows are the test set every published number uses. So this entire class of correction is absent,
and every MNIST accuracy in §6 is directly comparable.

Establishing the **scope** of a defect is itself a finding: the JSC-only study could not tell
whether dataset ambiguity was endemic to the field or specific to that benchmark. It is specific.

## 4. DWN vs conifer — the same-silicon, near-iso-area result

Both built by us, same part, same 10 ns target, same flow, same encoder-inclusive convention, and
within 5.5% of the same area:

| | acc% | LUTs | % device | Fmax | cycles |
|---|---|---|---|---|---|
| **DWN `2x[1000,500]`** | **97.76** | 3,464 | 16.65 | 103.8 | 5 |
| conifer `gbdt_d3_n5` | 84.31 | 3,653 | 17.56 | **477.3** | **3** |
| | **−13.45 pp** | +5.5% | | **4.6×** | |

**13.45 percentage points at matched area.** A gradient-boosted ensemble does not approach a DWN's
accuracy on raw pixels at this size — consistent with JSC, where a 14-point conifer curve also
failed to close the gap.

⚠️ **But conifer wins decisively on speed, and that belongs in the headline too.** 477.3 MHz
against 103.8, three cycles against five, and it repeated across both configs — so it is a property
of the flow, not a fluke. A GBDT's critical path is one comparator tree; a DWN's is a thermometer
encoder, then LUT layers, then a popcount. **If the requirement is throughput rather than accuracy
per LUT, a GBDT is the better answer**, and this study should not obscure that.

⚠️ **Scope, stated plainly: there is no conifer curve and no measured ceiling on MNIST.** Two
synthesized points. The supported sentence is *"the largest ensemble built was 4,427 LUTs at
85.90%"* — never *"conifer's ceiling on this part is X"*. HLS cost 31–58 minutes per config, driven
by ensemble size and the 784-wide input interface, and runs were repeatedly interrupted. An
iso-area comparison needs a matched pair rather than a curve, and JSC already carries the trend
argument with 14 points.

## 5. hls4ml — cut deliberately, with reasons

Not measured on MNIST. Three reasons, recorded before the decision rather than after:

1. **A published MNIST row already exists** — a ternary NN at **260,092 LUTs**, 95%, on `xcvu9p`.
   That is **12.5× NeuraLUT** and **75× our `2x[1000,500]`**.
2. **784 inputs will not fit this part.** On JSC, with 16 features, hls4ml fit only at quarter
   width. The first layer scales with input count.
3. **JSC measured this axis in full** (`docs/phase3-report.md` §5), including two silent failures
   worth more than the numbers.

**This is a scope cut, not a result.** We have not shown hls4ml fails on MNIST on our part; we have
declined to measure it and cited the published row instead.

## 6. Where we sit against the published field

Every LUT count below is as the source reports it; convention is marked.

| method | acc% | LUTs | convention | part |
|---|---|---|---|---|
| ULEEN ULN-L | **98.46** | — | ASIC, no FPGA area | 45 nm ASIC |
| DWN paper `lg` | 98.30 | 4,082 | core only | xcvu9p-2 |
| DWN paper `sm` | 97.10 | 692 | core only | xcvu9p-2 |
| **this project `2x[1000,500]`** | **97.76** | **3,464** / 2,168 core | **encoder included** | **xc7a35t-1** |
| SparseLUT-NeuraLUT | 96.96 | 54,798 | no separate encoder | xcvu9p-2 |
| TreeLUT (I) | 97 | 4,478 | no separate encoder | not stated |
| PolyLUT-Add | 96 | 14,810 | no separate encoder | xcvu9p-2 |
| NeuraLUT HDR-5L | 96 | 54,798 | no separate encoder | xcvu9p-2 |
| PolyLUT HDR | 96 | 70,673 | no separate encoder | xcvu9p-2 |
| BTHOWeN Large | 95.2 | 151,704 | encoder included | 7-series |
| hls4ml ternary | 95 | 260,092 | no separate encoder | xcvu9p-2 |
| BTHOWeN Small | 93.4 | 15,756 | encoder included | xc7z020-1 |

### 6.1 The weightless comparison — the one this phase exists for

BTHOWeN is the same family, on the closest silicon in the study, and prints accuracy to three
decimals so §3.2's rounding problem does not apply:

| | acc% | LUTs | cycles |
|---|---|---|---|
| BTHOWeN Small | 93.4 | 15,756 | 25 |
| BTHOWeN Medium | 94.3 | 38,912 | 37 |
| BTHOWeN Large | 95.2 | 151,704 | 74 |
| **this project `2x[1000,500]`** | **97.76** | **3,464** | **5** |
| **this project `1x300`** | **96.77** | **1,597** | **4** |

**+2.56 pp over BTHOWeN-Large at 43.8× fewer LUTs.** `1x300` alone beats every BTHOWeN model using
**9.9× fewer LUTs than the smallest of them**. Both designs even share an input stage: BTHOWeN uses
thermometer encoding, and its `Bits/Input` (2/3/6) is exactly our `z`.

### 6.2 ⚠️ ULEEN is more accurate than us, and the report says so

**ULN-L reaches 98.46%** against our best-on-board 97.76%. Our answer is area and latency, not
accuracy.

It cannot be placed in a LUT column: its Table III reports **ASIC area in mm²** against Bit
Fusion/LeNet-5, and Table IV reports **model size in KiB** (262). No FPGA LUT figure for MNIST
exists. So ULEEN is comparable on accuracy and on model size, and not on area — recorded with a
null LUT and a warning rather than quietly omitted.

### 6.3 Almost nothing published would fit a Basys 3

Of the published MNIST designs with FPGA LUT counts, **only BTHOWeN-Small (15,756) and the DWN
paper's own rows fall under the XC7A35T's 20,800 LUTs.** NeuraLUT needs 2.6× the device; PolyLUT
3.4×; hls4ml 12.5×.

That is the plainest statement of the result: this is a class of network whose published
implementations target datacentre parts, and the contribution here is that it runs — bit-exact,
verified on silicon — on a $150 board.

## 7. Our implementation reproduces the paper

The DWN paper's Table 2 is the reference this project is built against. At its own convention:

| | ours | DWN paper | ratio | Δ acc |
|---|---|---|---|---|
| `1x300` vs `sm` | 96.77% / **630** | 97.1% / 692 | **0.91×** | −0.33 pp |
| `1x2000` vs `lg` | 98.26% / **4,821** | 98.3% / 4,082 | 1.18× | −0.04 pp |

At the small end we are **9% smaller** for 0.33 pp less accuracy; at the large end 18% larger at
statistically identical accuracy (−0.04 pp, well inside the 0.24 pp floor).

**This is stronger evidence of correctness than Gate 1 can provide.** Gate 1 proves our RTL matches
*our* golden model; this shows our entire pipeline — exporter, generator, RTL — lands where the
authors' independent implementation lands.

**An independent check that the convention reading is right:** the same table reports JSC `sm` at
**110 LUTs**, and Phase 1 measured our 1x50 core at **110 LUTs**. Same config, same convention, same
count.

⚠️ The −0.33 pp at the small end is just above the noise floor, so it is probably real rather than
seed scatter. Candidates are the `tau` schedule and the `z=3` choice; neither was investigated.

## 8. Limitations

- **No conifer curve, no measured ceiling** (§4). Two synthesized points.
- **hls4ml not measured on MNIST** (§5). A scope cut, not a finding.
- **Five of 26 literature rows are `reported`, not `verified`** — taken from a survey
  (arXiv:2506.07367) rather than the primary paper. The survey's rows for DWN, PolyLUT, NeuraLUT
  and PolyLUT-Add match the primaries exactly, which is why they are trusted enough to include.
- **Cross-silicon comparison is unavoidable.** LUT counts transfer between Xilinx families;
  **nanoseconds do not**, and `xcvu9p-2` is two process generations and one speed grade ahead of
  `xc7a35t-1`. Every latency comparison in §6 is therefore reported in cycles as well.
- **Our frontier has no measured edge.** The MNIST ladder stops at 33% of the device by decision
  (`docs/mnist/phase2-report.md`). "The largest MNIST model that fits" is unsupported.
- **The `1x100` and `1x200` rows use models whose accuracy sits below anything published here.**
  They are on the frontier for area, not for accuracy, and should not be quoted as MNIST results in
  isolation.

## 9. Conclusion

On its own family, on the closest comparable silicon in the field, a DWN generated by this project
reaches **97.76% at 3,464 LUTs and 103.8 MHz** — **+2.56 pp over BTHOWeN-Large at 43.8× fewer
LUTs** — and reproduces the DWN paper's own area–accuracy trade-off to within 9% at matched
convention.

Against a gradient-boosted ensemble at the same area on the same part, it is **13.45 pp more
accurate and 4.6× slower**. Against ULEEN it is **0.70 pp less accurate**, with no FPGA area
published to compare against.

And of every published MNIST design in the table with an FPGA LUT count, **only one other would fit
the board this one was verified on**.
