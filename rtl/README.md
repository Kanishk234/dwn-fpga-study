# `rtl/` — the Verilog

Two kinds of thing live here, and the distinction matters:

| | |
|---|---|
| **`lut_node.v` `popcount.v` `argmax.v` `pipe_reg.v`** | **hand-written.** The only Verilog in this project written by a person. Every configuration instantiates them unchanged. |
| **`example-model-1x50/`** | **generated.** One trained model, emitted by `rtlgen/`, committed so it can be read. |

## The hand-written primitives

| file | what it does |
|---|---|
| `lut_node.v` | one DWN neuron = `assign out = TABLE[addr]` = **one LUT6**. The architectural premise (brief §4), and measured: 50 nodes synthesize to exactly 50 LUTs |
| `popcount.v` | per-class bit count — an adder tree, and the critical path at every width |
| `argmax.v` | picks the winning class from the scores |
| `pipe_reg.v` | one pipeline stage; `ENABLE=0` compiles it out entirely, which is how Group B sweeps depth |

`probe/` is a throwaway Phase 1a diagnostic — does `TABLE[addr]` really map to a single LUT6?
See `docs/reference/probe-results.md`. The whole area model rests on that answer.

## `example-model-1x50/` — a real model, in Verilog, committed to be read

The emitted RTL for one trained configuration: **50 LUT nodes, n=6, z=200,
`DistributiveThermometer`** — the paper's `sm`, and the only config this project has run on
silicon (Gate 1b: 166,000/166,000 on a Basys 3).

**It is a frozen sample, not live output.** Nothing regenerates into it and nothing builds from
it. It is here so you can see what the generator produces without installing Vivado. Its own
README walks through what to look at.

## Where the *live* generated RTL goes

Model-specific Verilog — LUT tables, wiring, encoder comparators — is a build product, emitted
per configuration:

```
build/rtl/                    default output
build/configs/<name>/rtl/     one sweep configuration
```

Both gitignored. `rtl/gen/` existed in Phase 1 and was removed in Phase 2: 54 configurations of
emitted Verilog do not belong in git. One committed example does.

To generate and verify one yourself:

```
.venv/Scripts/python.exe scripts/run_gate1.py
```

which emits into `build/rtl/` and then proves it bit-exact against the golden model.
