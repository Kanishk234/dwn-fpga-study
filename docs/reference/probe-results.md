# Phase 1a — LUT6 mapping probe results

**Run 2026-08-02. Vivado 2025.2, part `xc7a35tcpg236-1`. Result: risk #1 retired.**

The question: does a DWN LUT node written as `assign out = TABLE[addr]` map to exactly one Xilinx
LUT6 primitive? This is `project-brief.md` §12 risk #1, and it was checked before writing any other
code because it is a **branch point, not a task** — the RTL generator's output format, the DSE area
model, and the "one neuron is one LUT6" claim in §4 all depend on the answer.

Reproduce with:

```
python scripts/probe/gen_probe.py
vivado -mode batch -source scripts/probe/probe.tcl -notrace
```

---

## Results

| Variant | What it tests | Expected LUT6 | Got | Verdict |
|---|---|---|---|---|
| A `probe_a_baseline` | literal 64-bit `localparam`, `assign out = TABLE[addr]` | 37 | 37 | **PASS** |
| B `probe_b_param` | table width from a parameter, `[(1<<N)-1:0]` | 37 | 37 | **PASS** |
| C `probe_c_romstyle` | A + `(* rom_style = "distributed" *)` | 37 | 37 | **PASS** |
| D `probe_d_twolayer` | two layers, irregular inter-layer permutation | 60 (37+23) | 60 | **PASS** |

Every variant reported, from `report_utilization`:

```
LUT as Logic      37 (or 60)      <- all nodes, as intended
LUT as Memory      0              <- the failure mode risk #1 feared: did not occur
Slice Registers    0
Block RAM Tile     0
DSPs               0
```

Primitive counts were taken from `REF_NAME` on the synthesized netlist, **not** from the utilization
report's "Slice LUTs" line — that line rolls LUT1…LUT6 together and would have hidden a node landing
on a narrower primitive.

---

## What each result establishes

**A — the architectural premise holds.** One DWN-6 node becomes exactly one LUT6, and its 64 table
entries are free. Vivado did not infer distributed RAM, block RAM, or a multi-LUT decomposition.
`LUT as Memory = 0` confirms this directly rather than by inference from the total.

**B — `rtlgen` can be generic over `n` at no cost.** A parameterized table width synthesizes to an
identical netlist. This case became mandatory when `n` was made a Phase 2 sweep axis (CLAUDE.md,
brief §10); it turns out to need no special handling and no second codepath.

**C — do not emit `rom_style`.** It produces a byte-identical result. It was the documented
mitigation had A failed; A did not fail, so the attribute is noise in generated output. **`rtlgen`
should not emit it.**

**D — inter-layer wiring is genuinely free.** Two layers joined by an irregular,
learned-mapping-style permutation cost exactly 37 + 23 = 60 LUT6 with zero logic for the
permutation itself. This is the assumption `dse-plan.md` §5's area model rests on:

```
core LUTs ≈ W₁ + W₂ + ... + W_L + reduction cost
```

That model is now evidence-backed for the core, at small scale.

---

## What this does NOT establish

Two limits, both important, neither addressed by this probe:

**Routing congestion (risk #2) is completely untested.** This was **out-of-context synthesis only** —
no placement, no routing, no timing closure. Congestion is a place-and-route phenomenon and cannot
appear in a synthesis-only flow. Risk #2 remains the top technical risk, entirely open.

**Scaling behavior is untested.** These designs are 37–60 nodes. A real JSC config is hundreds to
thousands. Per-node mapping is established; how mapping and routability behave at 100× the node
count is not. Expect the first large synthesis to be informative.

**The thermometer encoder was not in scope here.** Risk #3 (encoder cost, up to 3.2× the core per
Mecik & Kumm) is untouched by this probe — no encoder was generated. That measurement comes with the
first end-to-end model.

---

## Consequences for the build

1. **Proceed with the planned node form.** `localparam [2**N-1:0] TABLE; assign out = TABLE[addr];`
   is correct and needs no attributes, pragmas, or restructuring.
2. **`rtlgen` emits `n` as a Verilog parameter**, not a literal — free, and required for Phase 2.
3. **No `rom_style` in generated output.**
4. **The DSE area model can be trusted enough to filter configs** before synthesis
   (`dse-plan.md` §5) — for the core. The encoder term in that formula is still unmeasured.
5. **Nothing here de-risks routing.** Keep brief §12 risk #2's mitigation: start from the smallest
   working model and scale up, and treat a routing failure as a measurement rather than a defect.

---

## Artifacts

- `scripts/probe/gen_probe.py` — generates the four variants. Tables are pseudo-random with a fixed
  seed, and the generator **rejects degenerate tables** (constant, or ignoring any input) — either
  would let synthesis optimize a node away and silently corrupt the count.
- `scripts/probe/probe.tcl` — per-variant OOC synthesis, primitive counting, summary table.
- `rtl/probe/*.v` — the generated RTL, committed as evidence for the numbers above.
- `build/probe/<variant>/{utilization.rpt,netlist.v}` — gitignored, regenerable.
