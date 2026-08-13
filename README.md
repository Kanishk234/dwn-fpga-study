# DWN on FPGA — a neural network with no math in it

**What if a neural network had zero multiplication, zero addition — nothing but memory lookups?**

That's a **Differentiable Weightless Neural Network (DWN)** ([Bacellar et al., ICML 2024](https://arxiv.org/abs/2410.11112)). Instead of the usual
weights-and-arithmetic every neural net uses, a DWN is built entirely out of tiny lookup tables. And
here's the part that makes it worth building in hardware: a lookup table is *also* the basic building
block an FPGA chip is made of. On this design, a "neuron" isn't just similar to a hardware lookup
table — it **is** one, 1-to-1.

The catch: the paper's authors released the training code, but never the hardware. So we're building
it from scratch — hand-writing the actual chip-level circuit (Verilog) and running it on a $150
student FPGA board, far smaller and cheaper than the chips the original paper used.

## What we're doing

1. **Build it.** Take a trained model, hand-write the digital circuit for it, and get it running on a
   physical board — not a simulation.
2. **Map its limits.** Sweep dozens of versions of the network — bigger, smaller, different settings —
   and find exactly how far you can push it before a small, cheap chip runs out of room.
3. **Compare it fairly.** Build the same task with the two standard industry tools for this kind of
   job, on the identical chip, and see how hand-built weightless hardware stacks up.

## Why it's interesting

- No arithmetic circuits at all — just memory, wired up by training instead of by hand.
- Runs in **nanoseconds** — fast enough for real-time particle physics triggers at the Large Hadron
  Collider, which is the actual benchmark task this project uses.
- Nobody has shown this running on hardware students can actually afford.

## Status

✅ **It runs on real hardware**, and we have mapped how far it goes.

| Phase | What | Status |
|---|---|---|
| 1 — Core | Get the model running on the board | ✅ **complete** |
| 2 — Design Space Exploration | Map how big/accurate/fast it can go | ✅ **complete** |
| 3 — Controlled Comparison | Compare against standard tools + published results | ✅ **complete** |
| Second dataset (MNIST) | Repeat all three on a different problem | ✅ **complete** |

> **Reproducing these numbers.** JSC figures are measured at the git tag **`jsc-complete`**, MNIST
> figures at **`mnist-complete`**. Later RTL improvements shift some JSC areas by a couple of
> lookup tables (≤0.12%), so `git checkout jsc-complete` reproduces the JSC figures exactly. The
> two sets describe different commits and should not be mixed in one table.

### Phase 1 — it works on silicon

The reference config is **`1x50`**: 50 lookup-table neurons, the paper's smallest JSC model.

| | |
|---|---|
| Hardware vs software | **166,000 / 166,000** exact |
| Accuracy | 73.84% (the paper's config: 74.0%) |
| The neural network | **108 LUTs** — the paper reports 110 |
| Whole design on the board | 2,058 LUTs (9.9% of the chip), 0 DSPs |
| Speed | 4 clock cycles, one result every clock — 99.5 M classifications/s |

### Phase 2 — how far it goes

**52 configurations**, every one verified bit-exact against the software model and
placed-and-routed on the real part.

![accuracy vs area](./docs/results/frontier.png)

| | |
|---|---|
| **Largest model that fits** | **`1x2400`** — 76.18%, 12,751 LUTs (**61%** of the chip) |
| **Best accuracy that fits** | `1x1600` variant — 76.35% at **66%** of the chip |
| The measured edge | timing, not area: two configs use <81% of the chip yet miss 100 MHz |
| Every config | **0 DSPs, 0 block RAM** |

**Four findings the original paper could not have seen**, because it reports three model sizes
and nothing between them:

1. **The paper's largest model runs on a $150 board.** Its 2400-neuron `lg` config was thought
   not to fit — we projected >100% of the chip ourselves. It uses **61%**, once you stop paying
   for a setting that buys nothing.
2. **The paper's thermometer setting is past its own knee.** It fixes `z=200` everywhere and
   never reports the cost. `z=50` gives up 0.24 points for **40% less silicon**, and `z=400`/`800`
   are *worse* while costing more.
3. **The encoder dominates at small sizes and inverts at large ones** — 14.1× the network at 50
   neurons, 2.8× at 2000. The input encoder is the real cost of a weightless network on a small
   FPGA, and published LUT counts routinely exclude it.
4. **What runs out first is the clock, not the chip.** Every model too big for the board still
   had a third of its area free — it just could not be clocked fast enough.

**Read this next:** [`docs/phase2-report.md`](./docs/phase2-report.md) — the full sweep, all 46
configurations, and the six things that broke along the way.

### Phase 3 — how it compares

Two halves: two competing toolchains — a **gradient-boosted decision tree** through conifer and a
**quantized neural network** through hls4ml — both synthesized on the *same part, same flow, same
clock* as every DWN config; and the published LUT-DNN literature, 32 rows across 11 methods.

![accuracy vs area on our dataset](./docs/results-cc/jsc-openml.png)

**Against a GBDT on identical silicon** — 14 conifer configurations, 10 of which fit:

| | |
|---|---|
| At any area budget | DWN is **+1.5 to +1.7 points** more accurate |
| At matched accuracy | DWN uses **2.3–6.0× fewer LUTs** |
| Can a GBDT catch up? | **No.** It tops out at 74.88% using 74% of the chip; DWN reaches that at 3,381 LUTs |
| Where the GBDT wins | **Speed, clearly** — 477 MHz against our 101, and 4–17 ns against our 27–40 |
| Both | 0 DSPs, 0 block RAM, every configuration |

**Against hls4ml on identical silicon** — 6 configurations, only 1 of which fits. The published
network needs **259,492 LUTs** on a 20,800-LUT chip, and is *still* 2.2× too big after 16× internal
time-sharing. Making it fit takes quarter-width layers, 12-bit numbers and 4× time-sharing:

| at ~8,700 LUTs | hls4ml | DWN |
|---|---|---|
| accuracy | 75.67%\* | **76.05%** |
| DSPs | **53** (of 90 on the chip) | **0** |
| block RAM | 2 | **0** |
| latency | 34 cycles | **4 cycles** |
| throughput | one result every 4 cycles | **one every cycle** |

Smaller, more accurate, no DSPs, and **8.5× lower latency at 4× the throughput**.

\**hls4ml's accuracy here is its full-precision figure — we could not measure the quantized one
without a C++ compiler, so the real number is lower and the comparison already favours it.*

**But the more useful finding is about the literature itself.** Setting up a fair comparison
turned out to be the hard part, because the standard comparison table in this field — the one
reproduced in the DWN paper, in the follow-up work, and in a 2025 survey — is broken in two ways:

1. **"JSC" is two different datasets.** An OpenML version (~830k samples) and a CERNBox version
   (~987k). They are routinely listed in one table, and the same method scores **~1.05 points
   higher** on OpenML — seven times our measurement noise. Half the numbers everyone compares
   against are on the other dataset.
2. **LUT counts mix conventions.** The DWN paper reports its largest model at 4,972 LUTs — the
   network only, no input encoder. With the encoder it is 7,011. Ours are always complete
   designs. Three different published numbers exist for that one model.

Both defects start in the primary sources and spread by copying. Our comparison scripts refuse
to plot two datasets on one axis, or to draw a frontier across two conventions — enforced in
code, not in a footnote.

*One caveat we are still chasing: we have asked the DWN authors which of the two JSC datasets
their published numbers use. Their paper says one thing and their released code says another. It
does not affect the conifer comparison above, but it does affect how our accuracy lines up with
theirs.*

**Where that leaves us, stated plainly:** on our own dataset the best published design reaches
76.0% in **1,780 LUTs** against our 12,751. We are not competitive on raw area with the
specialised LUT-DNN compilers. What is different about ours is that it includes the encoder, and
runs on a **$150 board** rather than a data-centre FPGA.

**Read this next:** [`docs/phase3-report.md`](./docs/phase3-report.md) — the comparison, both
literature defects, and what we chose not to measure.

### Second dataset — does the generator generalise, or did it learn one problem?

The same exporter and RTL generator, on a deliberately unfavourable target: **784 input features
instead of 16**, ten classes instead of five, and a natively 8-bit input where JSC's is continuous.
25 configurations, all place-and-routed.

![MNIST accuracy vs area, ours and published](./docs/results-cc-mnist/mnist.png)

| | |
|---|---|
| Hardware vs software | **10,000 / 10,000** exact, on the board |
| Best design meeting 100 MHz | **97.76%**, 3,464 LUTs (16.7% of the chip), 5 cycles |
| Smallest usable | 96.77% in **1,597 LUTs** (7.7%) |
| vs **BTHOWeN** — a weightless network of the same lineage, on comparable silicon | **+2.56 points at 43.8× fewer LUTs** |
| vs the DWN paper's own MNIST numbers | within **9%** at matched accounting convention |
| vs a boosted-tree ensemble at the same area on the same part | **+13.45 points**, but 4.6× slower |

**The port is the result, not the accuracy.** Seven places in the flow silently hard-coded a fact
about the first dataset — a feature count, a byte width, a class-index width. All are gone; dataset
facts now live in a descriptor, and adding a third dataset means adding a descriptor and nothing
else. Both datasets reproduce exactly from the same code.

Two things carry beyond MNIST. The **encoder-convention defect** recurs, and is *width-dependent*
here — the encoder is 72.3% of the design at 100 neurons and 20.9% at 2,000 — so no single
multiplier can repair a mis-conventioned table. And the **dataset-ambiguity defect does not apply**,
which establishes that it was specific to JSC rather than endemic to the field.

⚠️ Of every published MNIST design with an FPGA lookup-table count, **only one other would fit this
board.** NeuraLUT needs 2.6× the chip, PolyLUT 3.4×, hls4ml 12.5×.

**Read this next:** [`docs/mnist/report.md`](./docs/mnist/report.md) — the standalone MNIST study.

## Running it yourself

### What you need

| | |
|---|---|
| **Python 3.12** | pinned to match the Kaggle image that writes the checkpoints. 3.14 also works, and Phase 1 reproduces bit-for-bit on it, but the pin is deliberate; see `requirements.txt` |
| **Vivado 2025.2** | for anything that synthesizes. Simulation uses `xsim`, which ships with it |
| **A Basys 3** | *optional*. Everything except the board tests runs without one |
| **A GPU** | *only* for training new models. Upstream `torch_dwn` has no CPU path, so training runs on Kaggle (see [`training/README.md`](./training/README.md)). Every result here reproduces from committed checkpoints without training anything |

Vitis HLS is needed only for the hls4ml and conifer comparisons, and it lives inside the Vivado
install, so there is no second toolchain to set up.

```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Reproduce the headline claim

The one that matters: regenerate the hardware from a trained checkpoint and prove it bit-for-bit
identical to the software model. Exits non-zero if a single test vector disagrees.

```
.venv\Scripts\python.exe scripts\run_gate1.py
```

Then the full reproduction, 12 checks without a board and 22 with one. **Areas must match
exactly**, or this machine's numbers are not comparable to the ones reported here:

```
.venv\Scripts\python.exe scripts\verify_phase1.py
.venv\Scripts\python.exe scripts\verify_phase1.py --with-board
```

### On real hardware

```
.venv\Scripts\python.exe scripts\build_bitstream.py
.venv\Scripts\python.exe scripts\program.py            # volatile, lost on power cycle
.venv\Scripts\python.exe scripts\host.py --ping
.venv\Scripts\python.exe scripts\host.py --gate1b      # all 166,000 vectors on silicon
```

### One synthesis run

Out of context, reporting network, encoder and complete design separately. That split matters:
published figures for this architecture usually exclude the encoder, which is most of the cost at
small model sizes.

```
.venv\Scripts\python.exe scripts\run_synth.py          # synthesis only
.venv\Scripts\python.exe scripts\run_synth.py --impl   # place-and-route; the quotable numbers
```

### The design-space sweep

```
.venv\Scripts\python.exe dse\grid.py                   # what would be built, and the budget
.venv\Scripts\python.exe dse\run.py --all --impl       # the sweep, resumable
.venv\Scripts\python.exe dse\report.py                 # table, frontier, headline number
.venv\Scripts\python.exe dse\plot.py                   # figures
```

⚠️ **The full sweep is tens of hours of Vivado.** Every result is already committed under
[`docs/results/`](./docs/results/), so you do not need to re-run it to read or check the numbers.

### The comparisons

```
.venv\Scripts\python.exe cc\conifer\run_conifer.py --sweep    # boosted trees
.venv\Scripts\python.exe cc\hls4ml\run_hls4ml.py --shrink     # quantized MLP, shrink sequence
.venv\Scripts\python.exe cc\literature\table.py               # combined comparison table
.venv\Scripts\python.exe cc\literature\plot.py --snapshot     # both figures
```

## Documentation

**Start here:** [**`REPORT.md`**](./REPORT.md) — the full write-up as a standalone document.
Background, method, results, comparison, limitations, appendices and a glossary; readable without
any other file in this repository.

| | |
|---|---|
| [**`REPORT.md`**](./REPORT.md) | **the JSC study** — standalone, with appendices and glossary |
| [**`docs/mnist/report.md`**](./docs/mnist/report.md) | **the MNIST study** — the same, for the second dataset |
| [`docs/phase1-report.md`](./docs/phase1-report.md) · [`phase2-`](./docs/phase2-report.md) · [`phase3-`](./docs/phase3-report.md) | JSC, per phase: what was built, the sweep, the comparison |
| [`docs/mnist/`](./docs/mnist/) | MNIST, per phase — plus the learnable-reduction and noise-floor studies |
| [`docs/results/`](./docs/results/) · [`results-cc/`](./docs/results-cc/) | JSC measurements and comparison figures |
| [`docs/results-mnist/`](./docs/results-mnist/) · [`results-cc-mnist/`](./docs/results-cc-mnist/) | MNIST measurements and comparison figures |
| [`cc/literature/`](./cc/literature/) | published results as machine-readable tables, with per-row provenance |
| [`docs/project-brief.md`](./docs/project-brief.md) | the full technical plan |
| ledgers — [`docs/`](./docs/) and [`docs/mnist/`](./docs/mnist/) | dated working logs, including what was tried and retracted |

## Repository

| | |
|---|---|
| [`rtl/example-model-1x50/`](./rtl/example-model-1x50/) | **a real DWN in Verilog, committed to be read** — the config that ran on the board |
| `exporter/` | trained checkpoint → lookup tables, wiring, thresholds |
| `rtlgen/` | that export → Verilog |
| `rtl/` `tb/` | hand-written primitives · golden-model testbench |
| `harness/` | UART, vector store, benchmark FSM — the board design |
| `datasets/` | **per-dataset facts, as data** — the only place a feature count or word width lives |
| `dse/` | the sweep: grid, runner, area model, report, plots |
| `scripts/` | Gate 1, synthesis, bitstream, board host |
| `cc/` | the comparisons: conifer, hls4ml, and the published-literature table |
| `experiments/` | analyses outside the shipped flow |
| `training/` | Kaggle notebooks — training needs a GPU and runs off-machine |

## Built on

This project implements an architecture we did not invent. The DWN model, the training method,
and the idea that one weightless neuron maps to one FPGA lookup table are all from:

> **Differentiable Weightless Neural Networks**
> Bacellar et al., ICML 2024 — [arXiv:2410.11112](https://arxiv.org/abs/2410.11112) ·
> [PMLR v235](https://proceedings.mlr.press/v235/bacellar24a.html)

Training uses the authors' own implementation, [`alanbacellar/DWN`](https://github.com/alanbacellar/DWN),
vendored as a submodule pinned at `9f887a0`. Our exporter reads whatever checkpoint format that
commit produces — see [`docs/checkpoint-format.md`](./docs/checkpoint-format.md).

**What is ours:** the RTL and its generator, the golden-model testbench and bit-exactness harness,
the board design, the design-space exploration, and every measurement reported here. The upstream
repository ships PyTorch training only — no RTL, no HLS, no FPGA flow — which is the gap this
project exists to fill.

The benchmarks are **jet substructure classification (JSC)**, the standard low-latency FPGA-ML
task, via the `hls4ml_lhc_jets_hlf` dataset — and **MNIST**, on the canonical 10,000-sample test
split, as a second and deliberately different problem.

## Team

Two undergrads, one shared repo.
