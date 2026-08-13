# A real DWN, in Verilog — the Phase 1 reference config

Generated RTL for **`1x50`**: 50 LUT nodes, n=6, z=200, `DistributiveThermometer`, a single
learnable-mapped layer. This is the paper's `sm`, and the only configuration this project has
run on real silicon — Gate 1b passed **166,000/166,000** on a Basys 3.

**Frozen sample, not live output.** Nothing regenerates into this folder and nothing builds from
it. Everywhere else, emitted Verilog is a build product that lives in `build/` — `rtl/gen/` was
removed in Phase 2 because 54 configurations of it do not belong in git. One committed example
does: it lets you read what the generator produces without installing Vivado.

Reproduce it byte for byte:

```
.venv/Scripts/python.exe rtlgen/emit_core.py    training/artifacts/dwn_jsc_t200_distributive_50_l_b100_checkpoint.pt --outdir build/example
.venv/Scripts/python.exe rtlgen/emit_encoder.py training/artifacts/dwn_jsc_t200_distributive_50_l_b100_checkpoint.pt --outdir build/example
```

## The files

| file | what it is | measured area |
|---|---|---|
| `dwn_core.v` | 50 `lut_node` instances + 5 popcounts + argmax | **108 LUTs** |
| `thermometer_encoder.v` | 202 comparators, Q3.12 constants folded in | **1,519 LUTs** |
| `dwn_top.v` | wires encoder → core, carries the pipeline parameters | — |
| `dwn_core_params.vh`, `dwn_top_params.vh` | latency defines, read by the testbench and the harness so the two can never disagree about pipeline depth | — |

Post-route, out-of-context, `xc7a35tcpg236-1` at 10 ns: **1,619 LUTs (7.78% of the device),
269 FF, 0 BRAM, 0 DSP, 147.1 MHz, 4 cycles, II=1.**

## What to look at

**One neuron is one lookup table.** The learned weights are gone — what survives is a 64-bit
truth table and six wires:

```verilog
lut_node #(.N(6), .TABLE(64'hD8FF3EFFDDFED5AA))
    u_l0_n0 (.addr({x[3182], x[31], x[668], x[1990], x[1592], x[2865]}), .out(layer0[0]));
```

Those six indices come from `argmax` over the trained mapping's weight matrix. Training learns a
soft preference across all 3,200 input bits for every slot; inference keeps only the winner,
which is why a 43 MB checkpoint collapses into a few thousand integers.

⚠️ **Address bit order is load-bearing.** Slot 0 must land on `addr[0]`, so the concatenation is
emitted MSB-first. Reversed, the design still elaborates, still synthesizes, and is wrong on most
inputs — see `docs/reference/checkpoint-format.md` §2.

**The encoder is one comparator per selected threshold**, with the threshold folded to a Q3.12
integer at export time. The hardware does a signed compare and no arithmetic anywhere:

```verilog
bits[   0] = $signed(x_flat[0*16 +: 16]) > -$signed(16'sd8794);
```

Only **202 of the 3,200** thermometer bits are ever read; the rest are tied low and synthesis
removes them. That selection is why the encoder costs 1,519 LUTs rather than ~24,000.

**None of this was written by hand.** The only hand-written modules are `../lut_node.v`,
`../popcount.v`, `../argmax.v` and `../pipe_reg.v`, and every configuration instantiates them
unchanged.
