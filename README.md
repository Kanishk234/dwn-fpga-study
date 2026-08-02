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

🔧 Design complete, build starting. Nothing has touched real silicon yet.

| Phase | What | Status |
|---|---|---|
| 1 — Core | Get the model running on the board | not started |
| 2 — Design Space Exploration | Map how big/accurate/fast it can go | not started |
| 3 — Controlled Comparison | Compare against standard tools + published results | not started |
| Stretch — second dataset | Repeat 2 & 3 on a different problem | optional |

Full technical plan: [`project-brief.md`](./docs/project-brief.md)

## Team

Two undergrads, one shared repo.
