# A Weightless Neural Network on an Entry-Level FPGA, and Two Defects in How Its Field Compares Results

**Kanishk Sama · Krithik Sama**
The University of Texas at Austin

---

## Abstract

Differentiable Weightless Neural Networks (DWNs) replace the multiply-accumulate arithmetic of a
conventional neural network with lookup tables, which map directly onto the lookup tables an FPGA
is physically built from. Published DWN results target large data-centre FPGAs. We ask whether the
architecture survives on hardware a student can afford: a Digilent Basys 3, built around a Xilinx
Artix-7 XC7A35T with 20,800 lookup tables, roughly one fifty-seventh the logic of the device used
in the original work.

We built a hand-written Verilog implementation and a generator that compiles a trained checkpoint
into synthesizable hardware, verified every configuration bit-exactly against a software reference,
and deployed one to the board, where it classified all 166,000 test samples with zero mismatches.
We then swept 54 configurations through place-and-route to map the accuracy/area/latency frontier,
and compared the result against a gradient-boosted decision tree compiled to the same part through
the same synthesis flow.

The largest model in the original paper fits comfortably: 76.18% accuracy in 12,751 lookup tables,
61.3% of the device, using no DSP blocks and no block RAM. Against a boosted tree on identical
silicon, the weightless network is 1.5 to 1.7 percentage points more accurate at every area
budget, and needs 2.3 to 6.0 times fewer lookup tables at matched accuracy; the tree, however, runs
roughly four times faster.

Our most transferable finding is not a hardware measurement. In assembling the comparison we found
that the standard results table for this benchmark — reproduced in the original DWN paper, in
subsequent work, and in a 2025 survey — is unsound in two independent ways. First, "jet substructure
classification" refers to two different datasets whose accuracies differ by about 1.05 percentage
points, and published tables mix them freely. Second, reported area figures mix two conventions:
some include the input encoder and some omit it, and for one model three different published
figures exist. Both defects originate in primary sources and propagate by citation. We give the
corrected comparison and release the tooling that enforces the distinction.

---

## 1. Introduction

A neural network is normally a great deal of arithmetic. An FPGA, by contrast, is built from small
memories called lookup tables: a 6-input lookup table is a 64-bit memory that returns one bit for
each of the 64 possible input patterns. Implementing arithmetic on an FPGA means assembling it out
of these memories, which is indirect and expensive.

Weightless neural networks invert that relationship. Instead of learning multiplicative weights,
they learn the *contents of lookup tables* directly. A neuron is a table; inference is a memory
read. There is no multiplication anywhere in the network. When such a model is placed on an FPGA,
one learned neuron becomes one physical lookup table — the model and the hardware are the same
kind of object.

Differentiable Weightless Neural Networks [1] made this practical by showing how to train such
tables with gradient descent. The published results are strong, but they are demonstrated on a
Xilinx Virtex UltraScale+ VU9P, a data-centre part with roughly 1.18 million lookup tables. The
obvious question — whether the approach still works when the device is small — was open, and it is
not a question that theory answers, because on a small device the cost of preparing the *input* can
exceed the cost of the network itself.

This report answers it. Our contributions:

1. **A complete open implementation for a small FPGA.** Hand-written Verilog primitives plus a
   generator that turns a trained checkpoint into synthesizable hardware, with a verification
   procedure that proves bit-exactness against a software reference on every configuration. No
   comparable open implementation targeting small FPGAs existed.

2. **A design-space exploration the original work could not perform**, because it reports three
   model sizes and nothing in between. Fifty-four configurations, place-and-routed on the real
   part, mapping accuracy against area and latency — including the input encoder, which published
   figures routinely exclude.

3. **A controlled comparison against a gradient-boosted decision tree** on identical silicon
   through an identical synthesis flow, rather than across papers and parts.

4. **Two defects in the field's standard comparison table**, with corrected results and tooling
   that prevents the errors mechanically rather than by convention.

### 1.1 What we did not do

We did not measure a quantised multilayer perceptron compiled by hls4ml on our part. Its published
design needs three times our device's capacity, so it cannot fit unshrunk; shrinking it until it
fits is a separate experiment we did not run. Section 6.4 states the consequence, which is that our
comparison rests on one measured baseline rather than two.

---

## 2. Background

### 2.1 Lookup tables, in hardware and in the model

The Artix-7 XC7A35T contains 20,800 six-input lookup tables. Each stores 64 bits and, given six
input bits, returns the stored bit at that address. Everything else on the device — arithmetic,
control, comparison — is built from these.

A weightless neuron with *n* inputs is a table of 2^n learned bits, addressed by its inputs. For
*n* = 6 this is exactly the shape of the hardware primitive. We measured the correspondence
directly: a 50-neuron layer synthesizes to exactly 50 lookup tables. The mapping is one-to-one,
not approximate.

### 2.2 The architecture

A DWN processes data in four stages.

**Binarization.** Real-valued input features are converted to bits by *thermometer encoding*: each
feature is compared against a series of thresholds, producing a bit per threshold. A feature above
seven of two hundred thresholds yields seven ones followed by 193 zeros — a "thermometer" reading.
We use the distributive variant [2], where thresholds are placed at quantiles of the training
distribution rather than at even intervals, which is more accurate for skewed features.

**Lookup layers.** Groups of *n* bits address a table; each table emits one bit; those bits address
the next layer. Which bits feed which table is *learned* during training and then frozen, so the
wiring costs nothing at inference.

**Reduction.** The final layer's bits are divided equally among the output classes and counted —
a *popcount*, the number of ones in a group.

**Decision.** The class with the highest count wins, resolved by an argmax circuit.

No stage multiplies. The entire network is memory reads, one adder tree, and one comparison tree.

### 2.3 The encoder is not free

Thermometer encoding is usually described as preprocessing, but in a fully parallel FPGA design it
is hardware: one comparator per threshold that any neuron actually reads. With sixteen input
features and two hundred thresholds each, the encoder's output is 3,200 bits wide.

This matters more than it appears. In the smallest configuration we measured, the encoder occupies
1,519 lookup tables against the network's 108 — **fourteen times the model it feeds**. Published
figures for this architecture generally exclude it. An independent group reached the same
conclusion concurrently and measured the omission at up to 3.2× [3].

---

## 3. Implementation

### 3.1 Structure

Four Verilog modules are written by hand: the lookup-table neuron, the popcount adder tree, the
argmax comparator tree, and a pipeline register. Everything model-specific — table contents,
wiring indices, encoder thresholds — is generated from a trained checkpoint. A 50-neuron model
compiles to 50 table constants and 300 wire indices; transcribing those by hand would teach nothing
and risk a silent single-character error.

Input features are represented as 16-bit signed fixed-point numbers with 12 fractional bits, a
format written Q3.12: one sign bit, three integer bits, twelve fractional bits, covering [−8, +8).
Section 7 shows this choice was the most consequential unexamined parameter in the project.

### 3.2 Verification

**Nothing was considered working until it matched a software reference bit-for-bit on every test
vector, including edge cases.** This rule was applied without exception, including to code that
looked obviously correct.

The software reference quantizes exactly as the hardware does, so agreement is exact rather than
approximate, and any discrepancy is a real defect rather than rounding. Every configuration in this
report passed before any area figure was recorded.

Two experiences justify the strictness.

**A verification harness that cannot fail is not verification.** Our generator originally checked
its own output by re-deriving the expected tables with the same function that produced them. It
reported complete agreement while the design was wrong on 958 of 1,504 test vectors. A check must
be independent of the thing it checks.

**A packing bug invisible at the size we tested.** Table contents were packed using a library
routine that pads short byte sequences on the wrong side. For *n* ≥ 3 a table is a whole number of
bytes and nothing goes wrong. For *n* = 2 every entry landed at the wrong address. We had tested
only *n* = 6. Worse, the resulting failure would have looked exactly like an outcome we already
expected on independent grounds — the sort of coincidence that turns a bug into a "finding".

### 3.3 On real hardware

One configuration was deployed to a Basys 3: 50 neurons, *n* = 6, 200 thresholds per feature. Test
vectors were streamed over a serial link, classified on the board, and compared against the
software reference.

**All 166,000 test samples matched exactly.** The complete design, including the serial interface
and vector storage, occupies 2,058 lookup tables — 9.9% of the device — and uses no DSP blocks. The
network core is 108 lookup tables against the 110 reported in the original paper for the same
configuration.

---

## 4. Design-space exploration

### 4.1 Method

Fifty-four configurations were trained and put through synthesis and place-and-route on the target
part at a 10 ns clock, out of context (that is, without surrounding system logic, so measurements
reflect the design alone).

The counts are easy to conflate, so precisely: **54** configurations attempted, **52** built (two
were unbuildable for a reason given in Section 8), **51** fit within the device, and **42** of
those also reach the board's 100 MHz clock. A forty-first-versus-forty-second distinction is worth
one sentence: 41 met the exact timing target they were synthesized against, while one further
configuration was deliberately built at a more aggressive 125 MHz target, missed it by 0.02 ns, and
still runs well above 100 MHz. Area and accuracy figures below are drawn from the 52 built
configurations; usability on the board refers to the 42.

Axes swept: number of neurons (50 to 3,000); thresholds per feature (8 to 800); inputs per neuron
(2, 4, 6); encoding scheme; layer count; and pipeline depth.

**Measurement noise.** Repeating identical configurations gave a run-to-run spread of **0.15
percentage points**. We treat any accuracy difference below this as no difference, and say so
wherever it matters.

### 4.2 Results

**Figure 1.** Accuracy against area for every configuration measured on our device, alongside
published results on the same dataset. Marker fill distinguishes designs whose area includes the
input encoder from those reporting the network alone; the two are not interchangeable (Section 5.2).

![Accuracy against area, OpenML dataset](./docs/results-cc/jsc-openml.png)


| | |
|---|---|
| Largest model that fits | **76.18%**, 12,751 lookup tables (**61.3%** of device), 101.3 MHz, 4 cycles |
| Best accuracy that fits | 76.35% at 66.0% of device |
| Cheapest near-plateau | 75.95% at 33.6% of device |
| Every configuration | 0 DSP blocks, 0 block RAM |

Four findings follow.

**The original paper's largest model fits on a $150 board.** Its 2,400-neuron configuration was
expected not to — our own early estimate exceeded 100% of the device. It uses 61.3%. The estimate
was wrong because it assumed the paper's threshold count was necessary.

**The paper's threshold count is past its own knee.** The original work fixes 200 thresholds per
feature everywhere and never reports what that costs. Reducing to 50 gives up 0.24 percentage
points for **40% less silicon**. Increasing to 400 or 800 is *worse* on accuracy while costing
more. This is the parameter that decides whether the largest model fits, and it had never been
measured.

**The encoder dominates small models and stops dominating large ones.** The encoder-to-network
ratio falls from 14.1× at 50 neurons to 2.8× at 2,400. Conclusions drawn at one model size do not
transfer, which is a pattern we hit repeatedly (Section 8).

**What runs out is the clock, not the chip.** Every configuration too large for the board still had
a third of its area unused — it simply could not be clocked at 100 MHz. The binding constraint is
timing, and the critical path is the popcount adder tree, whose depth grows with layer width.

---

## 5. Comparing against published work is harder than it looks

This section is the report's most transferable contribution. Both problems below are properties of
the *literature*, not of our measurements, and both are still present in current papers.

### 5.1 "Jet substructure classification" is two datasets

The benchmark used throughout this field is jet classification from Large Hadron Collider data:
sixteen physics-derived features, five classes. There are two distributions of it.

| | OpenML version | CERNBox version |
|---|---|---|
| Source | `hls4ml_lhc_jets_hlf`, OpenML dataset 42468 | CERNBox LHC Jets |
| Samples | ~830,000 | 986,806 |
| Used by | DWN, TreeLUT, hls4ml — **and this work** | LogicNets, PolyLUT, PolyLUT-Add, NeuraLUT, AmigoLUT, ReducedLUT, SparseLUT |

The authors of NeuraLUT-Assemble state the situation plainly [4]:

> "Both datasets target the same jet classification task, but the CERNBox version contains 986806
> instances, while the OpenML version has about 830000. […] Experimentally, we observed that models
> trained on the OpenML dataset achieve higher accuracy."

**The gap is about 1.05 percentage points, and it can be measured cleanly** because two methods
report both. FPGN [5] gives NeuraLUT-Assemble at 76.0% on OpenML against 75.0% on CERNBox, and
itself at 76.0% against 74.9%. Two independent within-method measurements, both ~1.0.

That is **seven times our measurement noise**, and larger than almost every difference these papers
argue about. Yet the standard comparison table lists both kinds of row together. It does so in the
original DWN paper, in the follow-up encoder analysis, and in a 2025 survey of the field [6], which
divides the benchmark by accuracy band and never mentions that two data sources exist.

**Consequence.** The comparison most often quoted — DWN against LogicNets, PolyLUT and NeuraLUT —
is across datasets. Those are CERNBox results. On the OpenML data the honest peer group is smaller
and stronger: DWN, TreeLUT, NeuraLUT-Assemble, FPGN, and hls4ml.

### 5.2 Area figures mix two conventions

The original DWN paper reports its largest model at **4,972 lookup tables**. That figure excludes
the thermometer encoder. Including it, the same model measures **7,011** [3]. A third figure,
**6,302**, appears in an independent reimplementation [5] whose convention is not stated.

Three published numbers, one model, and nothing in the tables distinguishes them. Our figures are
always complete designs, so quoting ours against the 4,972 understates us by roughly the encoder's
entire cost.

**We verified the convention for the other architectures rather than assuming it.** They take
already-quantised inputs directly into table address lines and have no expansion stage, so the
question does not arise for them. TreeLUT quantises its inputs "as a pre-processing step" [7] —
off-chip, exactly as our feature scaling is. And the encoder analysis [3] states that it was DWN
specifically whose published evaluations "reported only the resource usage of the LUT layer and the
classification logic". DWN is the outlier.

### 5.3 Both defects begin upstream

The DWN paper's own results table lists hls4ml — an OpenML result — beside PolyLUT and NeuraLUT,
which are CERNBox results, and reports its own area core-only against their complete designs. Both
errors are present in the primary source and inherited by everything that cites it.

We therefore enforce the distinctions in software rather than in prose: our plotting tool refuses
to place two datasets on one axis, and refuses to draw a frontier across two area conventions. A
convention that must be remembered will eventually be forgotten; we made our own tools reject the
mistake. An early version of our own figure connected core-only points to encoder-included ones —
the very error the figure exists to expose — which is why the check is now mechanical.

---

## 6. Comparison results

### 6.1 Against a boosted decision tree, on identical silicon

Gradient-boosted decision trees are the standard alternative for tabular problems of this kind. We
trained fourteen, compiled them to hardware with conifer [8], and put them through **the same
synthesis flow, the same part, and the same clock target** as every weightless configuration. Ten
fit the device.

This is the comparison the field usually cannot make, because published numbers come from different
parts and different tool versions. Here only the model differs.

**At a fixed area budget, which model is more accurate:**

| Budget (lookup tables) | Weightless | Boosted tree | Difference |
|---|---|---|---|
| 4,000 | 75.27% | 73.64% | **+1.63 pp** |
| 8,000 | 75.95% | 74.36% | **+1.59 pp** |
| 12,751 | 76.18% | 74.50% | **+1.68 pp** |
| 20,800 (whole device) | 76.35% | 74.88% | **+1.48 pp** |

**At a fixed accuracy target, which model is smaller:**

| Target | Weightless | Boosted tree | Ratio |
|---|---|---|---|
| 73.6% | 1,619 | 3,774 | **2.3×** |
| 74.2% | 2,541 | 7,602 | **3.0×** |
| 74.5% | 2,541 | 15,363 | **6.0×** |
| ≥74.9% | 3,381 | never reaches it | — |

**A boosted tree does not reach the weightless network's accuracy on this device at any size that
fits.** The best fitting tree reaches 74.88% using 73.9% of the device. The weightless network
reaches that accuracy in 3,381 lookup tables and continues to 76.35%.

**Where the tree wins, and it is not close.** It closes timing at **477 MHz** where our design
manages 101, and its latency is 4.2–16.8 ns against our 27–40 ns. The reason is structural: our
critical path is a 2,400-input popcount tree, while a boosted tree is a shallow cascade of
comparisons. Measured in *clock cycles* the two are comparable — 2 to 8 against our 4 — which is
why latency should always be reported both ways. A design's nanosecond figure is a property of the
part it ran on as much as of the design.

**The fair summary is that the weightless network wins accuracy per unit area and the tree wins
speed.** Both configurations use zero DSP blocks and zero block RAM.

### 6.2 A conversion bug that would have invalidated every tree result

Worth recording because it produced no error message. Recent versions of the boosting library fit a
per-class base score that the hardware compiler could not read, emitting one class's initial value
as *not a number*. That single value made the class score undefined for every sample and rendered
the final comparison arbitrary — **127,034 of 166,000 predictions wrong** — while producing
perfectly plausible-looking hardware that synthesized without complaint.

It was caught by evaluating the compiler's own emitted model description with an independent
implementation: the same principle as our hardware verification, applied one level up.

### 6.3 Against published work, on our dataset

Restricted to the OpenML data, so the rows are comparable. DWN appears twice because two
conventions exist.

| Method | Accuracy | Lookup tables | Encoder included | Part |
|---|---|---|---|---|
| **This work** (largest fitting) | **76.35%** | 18,777 | yes | XC7A35T |
| DWN, as published | 76.3% | 4,972 | **no** | VU9P |
| DWN, encoder included [3] | 76.3% | 7,011 | yes | VU9P |
| **This work** (headline) | **76.18%** | **12,751** | yes | XC7A35T |
| hls4ml [9] | 76.0% | 63,251 | n/a | VU9P |
| TreeLUT [7] | 76.0% | 2,234 | n/a | VU9P |
| NeuraLUT-Assemble [4] | 76.0% | **1,780** | n/a | VU9P |
| FPGN [5] | 76.0% | 3,345 | unstated | VU9P |
| Boosted tree (this work) | 74.88% | 15,363 | n/a | XC7A35T |

**Read honestly: we are not competitive on raw area with the specialised compilers.** The best
published design reaches 76.0% in 1,780 lookup tables; ours needs 12,751.

Three things make the numbers less unlike than they appear, and all three should be stated whenever
they are quoted:

- **Ours include the encoder.** The 4,972 figure does not, and the encoder is a substantial
  fraction of a weightless design.
- **Ours run on a $150 board**, an Artix-7 at 100 MHz, not a data-centre part at 700 MHz. Area
  transfers approximately between devices; clock speed does not.
- **The row that shares both our dataset and our convention is DWN at 7,011.** That is the honest
  point of comparison, and against it our 12,751 reflects a smaller, slower device.

### 6.4 What is missing

We did not synthesize an hls4ml design on our part. Its published configuration needs 63,251
lookup tables against our 20,800 — three times over, which is arithmetic rather than an experiment.
What we therefore cannot report is **how much accuracy hls4ml retains when shrunk until it fits**,
which is a genuine open question.

This has a specific consequence. The "no DSP blocks" property is the sharpest distinction against
hls4ml, whose quantised networks spend 38 of them. But the boosted tree also uses none — trees do
not multiply either. **So on our own silicon that distinction is untested**; it rests on published
figures alone. A measurement of one shrunk hls4ml configuration would close this, and is in
progress at the time of writing.

---

## 7. The input word was six bits too wide

The concurrent encoder analysis [3] reports an encoder of 201 lookup tables where ours, for the
same model and the same threshold count, uses 1,519. Investigating a 7.6× discrepancy produced the
largest single area result in the project.

**The cause is comparator width, and nothing else.** That work normalises features to [−1, 1) and
quantises them to 6–9 bits. We carried a 16-bit word whose three integer bits exist only because
our features are standardised to roughly ±4.5, and whose twelve fractional bits are far finer than
the spacing between adjacent thresholds. Rebuilt at 8 bits, our encoder measures **182** lookup
tables against their 201. No structural technique was missing; the numbers were simply wider than
they needed to be.

We then measured accuracy and area across widths on the largest fitting configuration. Accuracy was
evaluated on all 166,000 test samples, not a subset.

| Input word | Encoder area | Reduction | Accuracy change | Acceptable? |
|---|---|---|---|---|
| 16 bits (as built) | 5,753 | — | — | — |
| 12 bits | 4,157 | 1.38× | −0.111 pp | yes |
| **11 bits** | **992** | **5.80×** | **−0.142 pp** | **yes** |
| 10 bits | 891 | 6.46× | −0.219 pp | no |
| 8 bits | 655 | 8.78× | −1.092 pp | no |

**There is a cliff between 12 and 11 bits** — a 4.7× area drop for one bit — where the comparator
stops needing a carry chain and collapses into plain lookup-table logic. The narrowest width that
preserves accuracy lands on the cheap side of that cliff **by exactly one bit**. Had the cliff sat
one bit lower, the usable saving would have been 1.38× instead of 5.80×. That is luck, and it means
neither limit can be extrapolated to another configuration without measuring both again.

Two further observations:

**The accuracy limit moves with model size, in the direction we did not expect.** On the small
model, accuracy survived down to 10 bits; on the large one, only to 11. Larger networks tolerate
*less* input precision, because each comparator feeds far more neurons — about 19 here against 1.5
in the small model — so a single wrong encoder bit propagates much further.

**Most of the saving needs no retraining.** A comparison is unchanged by any monotonic rescaling
applied to both the value and the threshold, so mapping each feature's threshold range onto the
available word is an export-time transformation, not a training change.

**We have not adopted it**, for a reason that also validates an earlier finding: the binding
constraint on this device is timing, not area, and the encoder is not on the critical path. Making
it six times smaller does not make a single additional configuration reachable. Projected, the
headline design would fall from 61.3% of the device to about 38%. That is worth having; it is not
worth destabilising a verified flow for at this stage.

---

## 8. Limitations and threats to validity

**One measured baseline, not two.** With hls4ml unmeasured (Section 6.4), the controlled comparison
rests on boosted trees alone.

**A dependency on an unanswered question.** The original DWN paper states that its jet-substructure
data follows NeuraLUT's, which would make it the CERNBox distribution — but its released code loads
the OpenML distribution, an independent reimplementation classifies it as OpenML, its accuracies
exceed every published CERNBox result, and our own reproduction on OpenML landed within 0.12
percentage points of its reported figure. Four lines of evidence against one sentence. **We have
asked the authors and have not yet received a reply.** If the answer is CERNBox, our accuracy
comparison against their largest model is across datasets and must be withdrawn; nothing else in
this report depends on it.

**A single dataset.** Every result here is jet substructure classification. Whether the generator
generalises is asserted by its parameterisation and not yet demonstrated on a second problem.

**Two configurations could not be built**, and the reason is a real limitation rather than a bug:
evenly-spaced thermometer thresholds span the full data range and exceed what our chosen
fixed-point format represents. A wider integer range would have fixed it at identical area, but the
encoding axis showed differences below our noise floor, so we recorded the failure instead of
extending the format.

**Datapath precision was fixed** across all 54 configurations, and Section 7 shows it was the
largest unexamined parameter in the project.

**Estimated figures are marked as such.** The projected area in Section 7 is derived from a measured
encoder but the complete design was not rebuilt at that width.

**A recurring failure mode, stated because it shaped the methodology.** Four times we drew a
conclusion from one configuration and generalised it: that the encoder cost a fixed multiple of the
network; that the reduction circuit was negligible; that input precision did not matter; that a
narrowing result validated on a subset held on the full test set. Each was corrected by measuring
at a second point. Any claim of the form "X is a small fraction of the design" should be read with
the range it was measured over attached.

---

## 9. Conclusion

A weightless neural network runs well on an entry-level FPGA. The largest configuration in the
original work fits in 61.3% of a device costing about $150, reaches 76.18% accuracy, uses no DSP
blocks and no block RAM, and matches its software reference exactly on all 166,000 test samples.
Against a boosted decision tree on identical silicon it is 1.5–1.7 percentage points more accurate
at every area budget and 2.3–6.0× smaller at matched accuracy, while being roughly four times
slower.

The parameter that decides whether the largest model fits is the number of thermometer thresholds,
which the original work fixes without reporting its cost; reducing it saves 40% of the silicon for
0.24 percentage points. What runs out first on this device is the clock, not the area.

But the finding most likely to matter to others concerns the comparison itself. The standard results
table for this benchmark mixes two datasets that differ by about 1.05 percentage points, and mixes
area figures that do and do not include the input encoder. Both defects begin in primary sources
and spread by citation. Any conclusion drawn from that table — including several of the comparisons
that motivated this project — needs rechecking against which dataset and which convention each row
actually uses.

---

## References

[1] A. Bacellar, Z. Susskind, M. Breternitz Jr., E. John, L. John, P. Lima, F. França.
*Differentiable Weightless Neural Networks.* International Conference on Machine Learning (ICML),
PMLR 235:2277–2295, 2024. arXiv:2410.11112. **Cite version 5**; version 1 reports different area
figures for the same configurations.

[2] A. Bacellar, Z. Susskind, L. Villon, I. Miranda, L. Araújo, D. Dutra, M. Breternitz Jr.,
L. John, P. Lima, F. França. *Distributive Thermometer: A New Unary Encoding for Weightless Neural
Networks.* ESANN, 2022.

[3] M. Mecik, M. Kumm. *Implementation and Analysis of Thermometer Encoding in DWN FPGA
Accelerators.* Asilomar Conference on Signals, Systems, and Computers, 2025. arXiv:2512.15251.

[4] M. Andronic, G. Constantinides. *NeuraLUT-Assemble: Hardware-aware Assembling of Sub-Neural
Networks for Efficient LUT Inference.* 2025. arXiv:2504.00592.

[5] J. Liang, H. Qin, et al. *FPGN: Redefining Ultra-Fast Programmable Gate-based Neural
Acceleration with Differentiable LUTs.* 2026. arXiv:2607.08427.

[6] *A Survey on LUT-based Deep Neural Networks Implemented in FPGAs.* 2025. arXiv:2506.07367.

[7] A. Khataei, K. Bazargan. *TreeLUT: An Efficient Alternative to Deep Neural Networks for
Inference Acceleration Using Gradient Boosted Decision Trees.* ACM/SIGDA International Symposium on
Field Programmable Gate Arrays (FPGA), pp. 14–24, 2025. arXiv:2501.01511.

[8] S. Summers et al. *conifer: Fast inference of Boosted Decision Trees in FPGAs for particle
physics.* Journal of Instrumentation 15, P05026, 2020. arXiv:2002.02534.

[9] F. Fahim et al. *hls4ml: An Open-Source Codesign Workflow to Empower Scientific Low-Power
Machine Learning Devices.* 2021. Figures as tabulated in [10].

[10] M. Andronic, G. Constantinides. *PolyLUT: Learning Piecewise Polynomials for Ultra-Low Latency
FPGA LUT-based Inference.* International Conference on Field Programmable Technology (ICFPT),
pp. 60–68, 2023. arXiv:2309.02334.

[11] Y. Umuroglu, Y. Akhauri, N. Fraser, M. Blott. *LogicNets: Co-Designed Neural Networks and
Circuits for Extreme-Throughput Applications.* International Conference on Field-Programmable Logic
and Applications (FPL), pp. 291–297, 2020.

[12] Z. Susskind, A. Arora, I. Miranda, A. Bacellar, L. Villon, R. Katopodis, L. Araújo, D. Dutra,
P. Lima, F. França, M. Breternitz Jr., L. John. *ULEEN: A Novel Architecture for Ultra-Low-Energy
Edge Neural Networks.* ACM Transactions on Architecture and Code Optimization 20(4), 2023.

---

## Appendix A — How configurations are named

A configuration is written **`W×N z=T`**, and where a value is omitted it takes the default shown.

| Element | Meaning | Default |
|---|---|---|
| `W` | number of lookup-table layers | 1 |
| `N` | neurons in each layer | — |
| `z=T` | thermometer thresholds per input feature | 200 |
| `n` | input bits per neuron | 6 |

So `1x2400 z=50` is one layer of 2,400 neurons, each reading 6 bits, with 50 thresholds per
feature. `1x50` is one layer of 50 neurons at the default 200 thresholds.

Boosted-tree configurations are written `gbdt_dD_nR`: maximum depth `D`, `R` boosting rounds. Since
the task has five classes, each round produces five trees, so `gbdt_d3_n40` is 200 trees of depth 3.

---

## Appendix B — Complete measurements

All figures are post-place-and-route on `xc7a35tcpg236-1` at a 10 ns clock, synthesized out of
context. Accuracy is on the full 166,000-sample test set.

### B.1 Weightless network, representative configurations

| Configuration | Accuracy | Network | Encoder | Total | Device | Clock | Cycles |
|---|---|---|---|---|---|---|---|
| `1x50` | 73.84% | 108 | 1,519 | 1,619 | 7.8% | 147.1 MHz | 4 |
| `1x200` | 75.32% | 466 | 4,570 | 5,036 | 24.2% | 113.9 MHz | 4 |
| `1x360 z=50` | 75.61% | 868 | 3,957 | 4,825 | 23.2% | 108.5 MHz | 4 |
| `1x800 z=50` | 75.95% | 1,893 | 4,970 | 6,981 | 33.6% | 104.9 MHz | 4 |
| `1x1200 z=50` | 76.05% | 2,875 | 5,384 | 8,444 | 40.6% | 102.3 MHz | 4 |
| **`1x2400 z=50`** | **76.18%** | 6,850 | 5,753 | **12,751** | **61.3%** | 101.3 MHz | 4 |
| `1x1600 z=100` | 76.35% | 4,153 | 9,260 | 13,729 | 66.0% | 101.8 MHz | 4 |

Full results for all 54 configurations, including those that did not fit, accompany this report as
machine-readable data.

### B.2 Boosted decision trees, all fitting configurations

| Configuration | Trees | Accuracy | Lookup tables | Device | Cycles | Clock |
|---|---|---|---|---|---|---|
| `gbdt_d3_n10` | 50 | 73.64% | 3,774 | 18.1% | 3 | 477.3 MHz |
| `gbdt_d4_n5` | 25 | 73.63% | 4,019 | 19.3% | 2 | 477.3 MHz |
| `gbdt_d3_n20` | 100 | 74.24% | 7,602 | 36.6% | 5 | 477.3 MHz |
| `gbdt_d5_n5` | 25 | 74.36% | 7,836 | 37.7% | 5 | 477.3 MHz |
| `gbdt_d4_n10` | 50 | 74.19% | 8,005 | 38.5% | 3 | 477.3 MHz |
| `gbdt_d6_n3` | 15 | 74.50% | 9,376 | 45.1% | 8 | 477.3 MHz |
| **`gbdt_d3_n40`** | 200 | **74.88%** | 15,363 | 73.9% | 7 | 477.3 MHz |
| `gbdt_d5_n10` | 50 | 74.77% | 15,605 | 75.0% | 6 | 477.3 MHz |
| `gbdt_d6_n5` | 25 | 74.80% | 15,898 | 76.4% | 8 | 477.3 MHz |
| `gbdt_d4_n20` | 100 | 74.75% | 16,052 | 77.2% | 5 | 477.3 MHz |

Four further configurations (75.09%–75.50%) exceeded the device and are excluded. All use zero DSP
blocks and zero block RAM.

### B.3 Encoder area against input word width

Measured on `1x2400 z=50`, 746 wired comparators, encoder synthesized alone.

| Input word | Lookup tables | Per comparator | Distinct thresholds |
|---|---|---|---|
| 16 bits | 5,753 | 7.71 | 745 / 746 |
| 13 bits | 4,916 | 6.59 | 739 |
| 12 bits | 4,157 | 5.57 | 734 |
| 11 bits | 992 | 1.33 | 731 |
| 10 bits | 891 | 1.19 | 720 |
| 9 bits | 794 | 1.06 | 687 |
| 8 bits | 655 | 0.88 | 596 |

### B.4 Published results, OpenML dataset only

Sources and per-row conventions accompany this report as machine-readable data. Rows on the CERNBox
distribution are deliberately excluded; see Section 5.1.

| Method | Accuracy | Lookup tables | Latency | Part |
|---|---|---|---|---|
| DWN `lg`, core only [1] | 76.3% | 4,972 | 7.3 ns | VU9P |
| DWN `lg`, with encoder [3] | 76.3% | 7,011 | 2.1 ns | VU9P |
| DWN `lg`, reimplemented [5] | 76.3% | 6,302 | 14.4 ns | VU9P |
| hls4ml [9] | 76.0% | 63,251 | 45.0 ns | VU9P |
| TreeLUT I [7] | 76.0% | 2,234 | 2.7 ns | VU9P |
| NeuraLUT-Assemble [4] | 76.0% | 1,780 | 2.1 ns | VU9P |
| FPGN [5] | 76.0% | 3,345 | 5.5 ns | VU9P |
| DWN `md`, with encoder [3] | 75.6% | 1,697 | 2.6 ns | VU9P |
| TreeLUT II [7] | 75.0% | 796 | 1.1 ns | VU9P |
| DWN `sm`, with encoder [3] | 74.0% | 311 | 2.0 ns | VU9P |

---

## Appendix C — Reproducing this work

Everything below is committed alongside this report. Training requires a GPU and was run off-machine;
everything else runs locally against a Xilinx toolchain.

| Step | What it does |
|---|---|
| Verify the implementation | Regenerates hardware from a checkpoint and proves it bit-exact against the software reference |
| Verify on the board | Streams all 166,000 test vectors to the device and compares every prediction |
| One synthesis run | Reports network, encoder and complete design separately |
| The full sweep | Runs verification, synthesis and place-and-route for every configuration; resumable |
| Comparison tables and figures | Regenerates everything in Sections 5 and 6 |

Two properties of the released data are worth noting. Trained checkpoints are not included — they
total roughly 933 MB — but every measurement derived from them is, in about 10 KB. And the raw
build outputs are regenerable but expensive, so the measurement snapshots are committed as the
evidence that each configuration was genuinely built rather than estimated.

---

## Glossary

**argmax** — the circuit that selects the highest-scoring class. Ties resolve to the lowest index.

**block RAM (BRAM)** — dedicated memory blocks on an FPGA, separate from lookup tables. All designs
here use none.

**bit-exact** — producing identical output to a reference implementation on every input, not merely
similar output. The verification standard used throughout.

**boosted decision tree (GBDT)** — an ensemble of decision trees trained in sequence, each
correcting its predecessors. The standard strong baseline for tabular data.

**critical path** — the slowest chain of logic between two registers. It sets the maximum clock
frequency.

**DSP block** — a hardware multiplier-accumulator on an FPGA. Weightless networks use none by
construction, which is the architectural point.

**FPGA** — a chip whose logic is configured after manufacture, built largely from lookup tables.

**Fmax** — the highest clock frequency at which a design meets timing.

**fixed-point, Q3.12** — a 16-bit signed number with three integer and twelve fractional bits,
representing values in [−8, +8) in steps of 1/4096.

**initiation interval (II)** — how many clock cycles must pass between successive inputs. II = 1
means one result per clock.

**iso-accuracy / iso-area** — comparing designs at matched accuracy (which is smaller?) or at
matched size (which is more accurate?), rather than comparing single points.

**latency** — the delay from input to result, given both in clock cycles and nanoseconds. Cycles
are a property of the design; nanoseconds also depend on the part.

**lookup table (LUT)** — the FPGA's basic logic element. A 6-input lookup table stores 64 bits and
returns one of them per input combination.

**out-of-context synthesis** — building a design without surrounding system logic, so measurements
describe the design alone. Standard practice for comparison, and used by all work cited here.

**Pareto frontier** — the set of configurations not beaten on both accuracy and size at once. The
meaningful output of a design-space exploration.

**percentage point (pp)** — an absolute difference between percentages. 76.2% versus 74.9% is 1.3
percentage points, not 1.7%.

**place-and-route** — assigning logic to physical locations and connecting it. Its timing figures
are trustworthy; post-synthesis estimates are systematically optimistic.

**popcount** — counting the ones in a group of bits. How class scores are computed here.

**RTL** — a hardware description at the level of registers and the logic between them; Verilog is
one such language.

**thermometer encoding** — converting a real value to bits by comparing it against a series of
thresholds, giving a run of ones followed by zeros. *Distributive* thermometer encoding places
thresholds at quantiles of the training data rather than at even intervals.

**weightless neural network (WNN)** — a network whose learned parameters are the contents of lookup
tables rather than multiplicative weights. Inference is memory lookup; there is no arithmetic.

**wiring (learned mapping)** — which encoder bits feed which neuron. Learned during training, then
frozen, so it costs nothing at inference.
