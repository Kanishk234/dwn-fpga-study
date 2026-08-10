# DWN on FPGA — a neural network with no math in it

**What if a neural network had zero multiplication, zero addition — nothing but memory lookups?**

That's a **Differentiable Weightless Neural Network (DWN)**, a 2024 ICML paper. Instead of the usual
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
| 3 — Controlled Comparison | Compare against standard tools + published results | next |
| Stretch — second dataset | Repeat 2 & 3 on a different problem | optional |

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

**Three findings the original paper could not have seen**, because it reports three model sizes
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

## Documentation

| | |
|---|---|
| [`docs/phase1-report.md`](./docs/phase1-report.md) | what was built, what broke, how to reproduce it |
| [`docs/phase2-report.md`](./docs/phase2-report.md) | the design-space exploration and its results |
| [`docs/results/`](./docs/results/) | every measurement, both figures, the trained grid |
| [`docs/project-brief.md`](./docs/project-brief.md) | the full technical plan |
| [`docs/phase1-ledger.md`](./docs/phase1-ledger.md) · [`phase2-ledger.md`](./docs/phase2-ledger.md) | dated working logs |

## Repository

| | |
|---|---|
| [`rtl/example-model-1x50/`](./rtl/example-model-1x50/) | **a real DWN in Verilog, committed to be read** — the config that ran on the board |
| `exporter/` | trained checkpoint → lookup tables, wiring, thresholds |
| `rtlgen/` | that export → Verilog |
| `rtl/` `tb/` | hand-written primitives · golden-model testbench |
| `harness/` | UART, vector store, benchmark FSM — the board design |
| `dse/` | the sweep: grid, runner, area model, report, plots |
| `scripts/` | Gate 1, synthesis, bitstream, board host |
| `experiments/` | analyses outside the shipped flow |

## Team

Two undergrads, one shared repo.
