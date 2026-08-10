# Phase 2 — the design-space exploration: what we swept, what we found, and what broke

Phase 2 is Study 1 of the project (brief §10): *what is the accuracy/area/latency Pareto frontier
for DWN on a fixed small FPGA?*

**Where Phase 1 sits in this.** Phase 1 built and verified exactly one configuration —
**`1x50`: n=6, z=200, `DistributiveThermometer`, a single learnable-mapped layer of 50 nodes**,
the paper's `sm`, at 73.84% and 1,619 LUTs. That is the sweep's first ladder rung and its only
config ever run on hardware (Gate 1b, 166,000/166,000). Phase 2 asks what happens when each axis
of that config is varied. It answers that with **46 configurations, every one Gate 1
verified and placed-and-routed on `xc7a35tcpg236-1`**, and it locates the device wall by
measurement rather than by extrapolation.

*(Six further configurations combining the width and `z` knees are queued for training; they are
noted in §7 and will extend §3.1 when measured.)*

`docs/phase2-ledger.md` is the raw dated log this was written from. `docs/results/` holds the
committed evidence: both figures, every measurement, and the grid as trained.

---

## 1. Headline results

| | |
|---|---|
| **Largest DWN measured to fit an XC7A35T** | **`1x2400 z=50`** — 2400 nodes, n=6, z=50 |
| **Its accuracy** | **76.18%** |
| **Its area** | **12,751 LUTs (61.30% of device)** = core 6,850 + encoder 5,753 |
| **BRAM / DSP** | **0 / 0** — at every one of 52 configs |
| **Timing** | 101.3 MHz, latency 4 cycles, II=1 |
| **Best accuracy that fits** | `1x1600 z=100` — **76.35%** at 66.0% of the device |
| **The measured edge** | area: `1x2000` at **102.8%**. Timing: `1x2400 z=100` and `1x3000 z=50`, both **96.2 MHz** |

**2400 nodes is the paper's `lg` — its largest JSC model.** It runs on a $150 Basys 3 using
under two-thirds of the chip. Phase 1 projected this exact configuration as *">100% of the
device, does not fit"*, from a ~24,000-LUT encoder estimate; measured at z=50 the encoder is
**5,753 LUTs**, a 4× error, and the whole design fits comfortably.

Four findings, in order of how much they change what is known:

1. **The paper's largest model fits, and the limit was never the network.** `lg` was thought
   impossible on this part. Cutting `z` from 200 to 50 — which costs 0.2 pp — makes it fit in
   61% of the device at 101.3 MHz.
2. **The binding constraint is timing, not area.** Everything past ~2400 nodes has area to
   spare and no clock headroom: `1x3000 z=50` uses 67% of the device and misses 100 MHz;
   `1x2400 z=100` uses 80% and misses. Four pipeline stages is the architectural maximum for a
   single-layer model, so there is no fix within this RTL.
3. **z=200 is past its own knee, and this is what makes (1) possible.** z=50 gives up 0.24 pp
   for 40% less silicon; z=400 and z=800 are *worse than z=200 while costing more*. The paper
   fixes z=200 for every JSC config and never reports what it costs.
4. **Accuracy saturates around 600 nodes.** `1x600` scores 76.10% at 51% of the device — within
   0.25 pp of the best config measured, against a 0.15 pp run-to-run noise floor. The
   encoder/core ratio also **inverts** across the ladder, 14.1× → 2.8×: past ~1200 nodes the
   *core* is the growth term, reversing Phase 1's central finding.

---

## 2. What the sweep covered

37 configurations from `dse/grid.py`, following `docs/dse-plan.md` §6's structure:

| group | axis | values | configs |
|---|---|---|---|
| **ladder** | layer width | 50, 100, 200, 360, 500, 600, 800, 1200, 1600, 2000 | 10 |
| **one-factor** | `z` | 8, 25, 50, 100, 400, 800 (baseline 200) | 12 |
| | encoding | gaussian, linear (baseline distributive) | 4 |
| | `n` | 4, 2 (baseline 6) | 4 |
| | layer count | 2, 3 (baseline 1) | 4 |
| **Group B** | pipeline depth, clock | 3-stage ×2, 2-stage, 8 ns, 12 ns | 5 |

One-factor axes vary from a fixed baseline at two mid-ladder rungs (200 and 360 nodes), so each
comparison is clean. Group B needs **no retraining** — same weights, different hardware — which
is why 37 configs required only 34 training runs.

**Cost:** 34 Kaggle GPU training runs (~10 h across two accounts), and ~3 h of serial Vivado for
place-and-route of all 36 buildable configs.

---

## 3. Complete results

All post-route, out-of-context, `xc7a35tcpg236-1`, constrained at 10.0 ns unless stated.
**Core and encoder are reported separately, always** (brief §6) — a combined total is what makes
the encoder's cost invisible in the literature.

### 3.1 Group A — model architecture

Each config is its own trained model. These are the points the accuracy/area frontier is built
from.

| config | axis | acc % | core | encoder | top | % dev | Fmax | cyc | lat ns |
|---|---|---|---|---|---|---|---|---|---|
| **`1x50`** † | ladder | 73.84 | 108 | 1519 | 1619 | 7.78 | 147.1 | 4 | 27.2 |
| `1x100` | ladder | 74.81 | 206 | 2608 | 2814 | 13.53 | 139.9 | 4 | 28.6 |
| `1x200` | ladder | 75.32 | 466 | 4570 | 5036 | 24.21 | 113.9 | 4 | 35.1 |
| `1x360` | ladder | 75.85 | 868 | 7138 | 8006 | 38.49 | 104.6 | 4 | 38.2 |
| `1x500` | ladder | 76.04 | 1118 | 8432 | 9542 | 45.88 | 110.6 | 4 | 36.2 |
| **`1x600`** | ladder | **76.10** | 1312 | 9367 | 10631 | **51.11** | 104.2 | 4 | 38.4 |
| `1x800` | ladder | 76.20 | 1878 | 10866 | 12904 | 62.04 | 100.3 | 4 | 39.9 |
| `1x1200` | ladder | 76.30 | 2868 | 12824 | 16061 | 77.22 | 100.7 | 4 | 39.7 |
| **`1x1600`** | ladder | **76.35** | 4124 | 14316 | 18777 | **90.27** | 103.2 | 4 | 38.8 |
| `1x2000` ❌ | ladder | 76.43 | 5496 | 15538 | 21382 | **102.80** | 94.9 | 4 | 42.1 |
| `1x200 z=8` | z | 72.80 | 466 | 940 | 1406 | 6.76 | 125.0 | 4 | 32.0 |
| `1x200 z=25` | z | 74.75 | 466 | 2075 | 2541 | 12.22 | 109.9 | 4 | 36.4 |
| `1x200 z=50` | z | 75.21 | 466 | 2924 | 3390 | 16.30 | 118.8 | 4 | 33.7 |
| `1x200 z=100` | z | 75.25 | 466 | 3692 | 4158 | 19.99 | 114.8 | 4 | 34.8 |
| `1x200 z=400` | z | 75.42 | 466 | 5439 | 5905 | 28.39 | 121.8 | 4 | 32.8 |
| `1x200 z=800` | z | 75.30 | 466 | 6023 | 6489 | 31.20 | 125.7 | 4 | 31.8 |
| `1x360 z=8` | z | 73.11 | 868 | 954 | 1822 | 8.76 | 105.8 | 4 | 37.8 |
| `1x360 z=25` | z | 75.27 | 868 | 2513 | 3381 | 16.25 | 109.4 | 4 | 36.6 |
| `1x360 z=50` | z | 75.61 | 868 | 3957 | 4825 | 23.20 | 108.5 | 4 | 36.9 |
| `1x360 z=100` | z | 75.75 | 868 | 5553 | 6421 | 30.87 | 113.9 | 4 | 35.1 |
| `1x360 z=400` | z | 75.81 | 868 | 8474 | 9334 | 44.88 | 109.4 | 4 | 36.6 |
| `1x360 z=800` | z | 75.77 | 868 | 9704 | 10572 | 50.83 | 110.3 | 4 | 36.3 |
| `1x200 gaussian` | encoding | 75.33 | 466 | 5069 | 5535 | 26.61 | 115.0 | 4 | 34.8 |
| `1x360 gaussian` | encoding | 75.73 | 868 | 7555 | 8423 | 40.50 | 104.6 | 4 | 38.2 |
| `1x200 n=2` | n | 74.06 | 450 | 1910 | 2319 | 11.15 | 124.5 | 4 | 32.1 |
| `1x200 n=4` | n | 75.03 | 466 | 3352 | 3795 | 18.25 | 126.8 | 4 | 31.5 |
| `1x360 n=2` | n | 74.63 | 852 | 2839 | 3635 | 17.48 | 117.7 | 4 | 34.0 |
| `1x360 n=4` | n | 75.47 | 867 | 5294 | 6129 | 29.47 | 115.2 | 4 | 34.7 |
| `2x100` | layers | 74.42 | 299 | 2820 | 2956 | 14.21 | **155.5** | 5 | 32.2 |
| `3x65` | layers | 73.87 | 273 | 2123 | 2365 | 11.37 | 148.1 | 6 | 40.5 |
| `2x180` | layers | 75.10 | 559 | 4410 | 4281 | 20.58 | 114.3 | 5 | 43.7 |
| `3x120` | layers | 74.49 | 516 | 3271 | 3658 | 17.59 | 152.3 | 6 | 39.4 |

| `1x800 z=50` | width×z | 75.95 | 1893 | 4970 | 6981 | 33.56 | 104.9 | 4 | 38.1 |
| `1x1200 z=50` | width×z | 76.05 | 2875 | 5384 | 8444 | 40.60 | 102.3 | 4 | 39.1 |
| `1x1600 z=100` | width×z | 76.35 | 4153 | 9260 | 13729 | 66.00 | 101.8 | 4 | 39.3 |
| **`1x2400 z=50`** | width×z | **76.18** | 6850 | 5753 | **12751** | **61.30** | 101.3 | 4 | 39.5 |
| `1x2400 z=100` ⚠️ | width×z | 76.39 | 6988 | 10084 | 16681 | 80.20 | 96.2 | 4 | 41.6 |
| `1x3000 z=50` ⚠️ | width×z | 76.16 | 8230 | 5887 | 13972 | 67.17 | 96.2 | 4 | 41.6 |

❌ exceeds the device · ⚠️ misses the 100 MHz board clock · † **the Phase 1 reference config**

**† `1x50` is Phase 1.** Everything Phase 1 built and verified is this one configuration —
**n=6, z=200, `DistributiveThermometer`, a single learnable-mapped layer of 50 nodes**, which is
the paper's `sm`. Its 1,619 LUTs, 147.1 MHz and 166,000/166,000 Gate 1b run on real silicon are
the anchor the whole sweep extends from, and it is the only config that has ever been on a
board. The sweep re-measured it and reproduced 108 / 1519 / 1619 exactly.

**Every config: 0 BRAM, 0 DSP.** Omitted from the tables because the value never varies — which
is the point. That column is the claim against hls4ml, whose quantized MLPs spend DSPs on
multiply-accumulate, and it is now measured across 46 designs rather than observed once.

### 3.2 Group B — hardware variants of one trained model

No retraining: same weights, same accuracy, different register placement or clock target. So
accuracy is constant within a rung and the interesting columns are timing and latency. Extended
2026-08-09 from one rung to four, because a single measurement had been stated as a general
result.

| variant | pipe | clk ns | top LUTs | FF | Fmax | WNS | cyc | lat ns | meets 100 MHz |
|---|---|---|---|---|---|---|---|---|---|
| `1x50 3-stage, no POP` | 1101 | 10 | 1623 | 249 | 107.5 | +0.699 | 3 | 27.9 | ✅ |
| `1x50 3-stage, no OUT` | 1110 | 10 | 1619 | 266 | 113.9 | +1.221 | 3 | **26.3** | ✅ |
| `1x50 2-stage` | 1100 | 10 | 1631 | 246 | 101.6 | +0.161 | 2 | **19.7** | ✅ |
| `1x360 3-stage, no OUT` | 1110 | 10 | 8010 | 1323 | 99.4 | −0.059 | 3 | 30.2 | ❌ |
| `1x360 3-stage, no POP` | 1101 | 10 | 8200 | 1300 | 72.5 | −3.793 | 3 | 41.4 | ❌ |
| `1x360 2-stage` | 1100 | 10 | 8215 | 1302 | 66.5 | −5.040 | 2 | 30.1 | ❌ |
| `1x360 clock 8ns` | 1111 | 8 | 8037 | 1326 | 124.7 | −0.020 | 4 | 32.1 | ❌ |
| `1x360 clock 12ns` | 1111 | 12 | 8006 | 1326 | 102.1 | +2.207 | 4 | 39.2 | ✅ |
| `1x600 3-stage, no OUT` | 1110 | 10 | 10641 | 1845 | 102.7 | +0.265 | 3 | **29.2** | ✅ |
| `1x600 3-stage, no POP` | 1101 | 10 | 10993 | 1825 | 64.5 | −5.505 | 3 | 46.5 | ❌ |
| `1x600 2-stage` | 1100 | 10 | 10987 | 1814 | 61.3 | −6.300 | 2 | 32.6 | ❌ |
| `1x1600 3-stage, no OUT` | 1110 | 10 | 18762 | 3499 | 100.9 | +0.086 | 3 | **29.7** | ✅ |
| `1x1600 3-stage, no POP` | 1101 | 10 | 18796 | 3460 | 58.6 | −7.069 | 3 | 51.2 | ❌ |
| `1x1600 2-stage` | 1100 | 10 | 18794 | 3462 | 50.7 | −9.739 | 2 | 39.4 | ❌ |

`pipe` is `ENC/LUT/POP/OUT` — which of the four register stages are enabled.

### Not buildable

| config | accuracy | why |
|---|---|---|
| `1x200 linear` | 75.22 | thresholds exceed Q3.12's ±8 range — see §5.3 |
| `1x360 linear` | 75.78 | same |

---

## 4. What the axes say

### 4.1 Width — accuracy saturates well before the device does

```
50   73.84  <- Phase 1 (7.8% dev)    600   76.10  (51% dev)
100  74.81                          800   76.20
200  75.32                         1200   76.30
360  75.85                         1600   76.35  (90% dev)
500  76.04                         2000   76.43  (103% dev -- does not fit)
```

From `1x600` to `1x1600` — a 2.7× increase in nodes and 39 points of device occupancy —
accuracy moves **0.25 pp**, against a 0.15 pp noise floor. The last four rungs are within noise
of each other.

**`1x1200` matches the paper's `lg` (1×2400) at 76.3%, using half the width.** The paper reports
`sm`/`md`/`lg` and nothing between, so the saturation point was not visible to it.

### 4.2 `z` — the paper's unexamined constant, costed

At 1×360:

| z | 8 | 25 | 50 | 100 | **200** | 400 | 800 |
|---|---|---|---|---|---|---|---|
| accuracy | 73.11 | 75.27 | 75.61 | 75.75 | **75.85** | 75.81 | 75.77 |
| LUTs | 1,822 | 3,381 | 4,825 | 6,421 | **8,006** | 9,334 | 10,572 |

The knee is at **z≈50**. Beyond it, z buys ~0.2 pp for 66% more area; past z=200 it buys
*nothing at all* — z=400 and z=800 score lower while costing more, both differences inside the
noise floor.

`z` drives comparator count, comparators are ~90% of the design, and the paper fixes z=200 for
every JSC configuration without reporting its cost. **This is the axis nobody has swept.**

### 4.3 `n` — n=2 is on the frontier, contradicting the plan

`dse-plan` §3 predicted n=2 and n=4 would be "worse on both axes." Measured at 1×200:

| n | accuracy | LUTs |
|---|---|---|
| 6 | 75.32 | 5,036 |
| 4 | 75.03 | 3,795 |
| **2** | **74.06** | **2,319** |

n=2 is 1.26 pp worse but **2.2× cheaper**, and is not dominated — it is a legitimate Pareto
point. The mechanism: fewer wiring slots select fewer distinct thresholds, which shrinks the
encoder faster than accuracy falls. The prediction assumed the *core* would dominate; on this
part the encoder does.

### 4.4 Layers — single layer wins on accuracy, multi-layer wins on speed

At ~200 nodes: `1x200` 75.32% > `2x100` 74.42% > `3x65` 73.87%, matching the paper's
all-single-layer JSC configs.

But `2x100` reached **155.5 MHz** and `3x120` 152.3 — the fastest designs in the sweep, against
113.9 MHz for `1x200`. `PIPE_LUT` inserts a register *per layer*, so depth buys pipelining for
free. A multi-layer model trades accuracy for timing headroom, which matters because **4 stages
is the architectural maximum for a single-layer design**.

### 4.5 Encoding — no measurable effect

At 1×360: distributive 75.85, gaussian 75.73, linear 75.78. A **0.12 pp** spread, below the
0.15 pp noise floor. On JSC, the choice of thermometer does not matter for accuracy — but it
does for representability (§5.3).

### 4.6 Group B — drop the output register, never the popcount one

Pipeline depth was swept at four rungs spanning the full slack range. `Fmax`, and whether it
meets the board's 100 MHz:

| rung | slack at 4 stages | no OUT reg | no POP reg | 2-stage |
|---|---|---|---|---|
| `1x50` | +3.200 ns | **113.9 ✅** | **107.5 ✅** | **101.6 ✅** |
| `1x360` | +0.440 ns | 99.4 ❌ | 72.5 ❌ | 66.5 ❌ |
| `1x600` | +0.400 ns | **102.7 ✅** | 64.5 ❌ | 61.3 ❌ |
| `1x1600` | +0.310 ns | **100.9 ✅** | 58.6 ❌ | 50.7 ❌ |

**The two 3-stage variants are not interchangeable, and the gap grows with width.** Removing the
argmax register costs 6 MHz at `1x50` and 2 MHz at `1x1600`. Removing the popcount register
costs 6, 27, 38 and **42 MHz** as width rises. The popcount is an adder tree whose depth grows
with layer width, and it is the critical path — the argmax is a small comparison over five
scores and is nearly free to un-register.

So the design rule is unambiguous: **to cut latency, drop the output register. Never the
popcount one.**

**This buys real latency on the configs that matter.** `1x1600`, the largest that fits, runs at
3 stages and **29.7 ns instead of 38.8 — 23% lower latency, 15 fewer LUTs, still meeting the
board clock.** `1x600` likewise, at 29.2 ns.

⚠️ **Two of these margins are inside placement noise.** `1x1600` at 3 stages passes with
**+0.086 ns** and `1x360` fails with **−0.059 ns**. Phase 1 measured placement varying by tens
of picoseconds between runs, so neither should be quoted as settled without a repeat. `1x360`
failing while `1x600` passes is not a trend — it is two configs either side of a knife edge.

**Latency must be ranked in nanoseconds, not cycles.** At `1x360`: 4 stages → 38.2 ns, 3 stages
→ 30.2 ns, 2 stages → 30.1 ns. Removing the third register buys nothing — the Fmax loss exactly
cancels the cycle saved. Ranked on cycles it looks like a win; ranked on Fmax, a loss. Both are
wrong.

⚠️ **Correction, kept visible.** This section previously read *"no reduced-pipeline variant meets
the board clock"*. That was measured at `1x360` alone and stated as a general result. Extending
to four rungs showed it false at three of them — at `1x50` **every** variant passes, including
2-stage at 19.7 ns, half the baseline latency.

### 4.7 The two knees compound — and that is what makes the paper's `lg` fit

One-factor-at-a-time has a structural blind spot: every non-baseline axis value was tested only
at widths 200 and 360, so **no pair of non-baseline values was ever tried**. The sweep found two
independent knees — width saturates around 600 nodes, `z` around 50 — and never asked what
happens when both are taken.

Six configs were added to close that. Every predicted accuracy landed within **0.10 pp**:

| config | accuracy | LUTs | % dev | Fmax | |
|---|---|---|---|---|---|
| `1x800 z=50` | 75.95 | 6,981 | 33.6 | 104.9 | |
| `1x1200 z=50` | 76.05 | 8,444 | 40.6 | 102.3 | |
| `1x1600 z=100` | **76.35** | 13,729 | 66.0 | 101.8 | best accuracy that fits |
| **`1x2400 z=50`** | 76.18 | **12,751** | **61.3** | 101.3 | **the headline** |
| `1x2400 z=100` | 76.39 | 16,681 | 80.2 | 96.2 | ⚠️ misses the clock |
| `1x3000 z=50` | 76.16 | 13,972 | 67.2 | 96.2 | ⚠️ misses the clock |

**The axes compound because they cost in different places.** Width buys accuracy and spends
*core* LUTs; `z` spends *encoder* LUTs and, past ~50, buys almost nothing. The encoder is the
larger term at every size the paper reports — so cutting `z` frees the budget to buy width.

Two comparisons make it concrete:

- **`1x1600 z=100` vs `1x1600`** — *identical* accuracy (76.35%), **13,729 vs 18,777 LUTs**.
  27% less silicon for literally no accuracy cost.
- **`1x2400 z=50` vs the Phase 1 projection** — `lg` was written off at ">100% of the device"
  on a ~24,000-LUT encoder estimate. Measured, its encoder is **5,753 LUTs** and the design
  occupies **61.3%**.

**And the constraint changed identity.** Every config that now fails, fails on *timing* with
area to spare: `1x3000 z=50` at 67% of the device and 96.2 MHz, `1x2400 z=100` at 80% and
96.2 MHz. Four stages is the architectural maximum for a single-layer model (§7), so neither has
a remedy in this RTL — though a multi-layer model of the same node count would gain a fifth
stage *and* a smaller encoder, since only the first layer reads thermometer bits.

⚠️ **The two failures land on the same 96.2 MHz.** With only two points that is as likely to be
coincidence as a structural limit, and it is not investigated here.

## 5. The things that actually cost time

### 5.1 The area model was wrong by 2×, and it silently excluded two configs

`1x600` was predicted at **96.9%** of the device and measured at **51.11%**.

The cause was a constant **67% selection ratio** — the fraction of wiring slots resolving to
distinct thermometer bits — measured at `sm` alone. Across the sweep the true ratio falls
monotonically with width: 67% at `1x50`, 51% at `1x200`, 44% at `1x360`, **25% at `1x1200`**, as
more slots compete for the same `features × z` bits.

**That mis-prediction filtered `1x800` and `1x1200` out of the sweep entirely**, at a predicted
128% and 133%. Both were re-run afterwards; both fit comfortably (62% and 77%).

Replaced with an **occupancy model**: `S` slots drawing from `M` bits yield `M(1-(1-1/M)^S)`
distinct bits if independent, scaled by a concentration factor `c(S/M)` — fitted as a quadratic
in `log10(S/M)`, because the learnable mapping concentrates on informative features rather than
drawing uniformly.

| | old (constant ratio) | new (occupancy) |
|---|---|---|
| worst error, comparators | **+110%** | **11.1%** |
| mean error, comparators | ~35% | **4.0%** |
| worst error, `dwn_top` area | >100% | 17.7% |

Fitted on 30 configs spanning widths 50–1200, n 2/4/6, z 8–800, 1–3 layers.

**The general lesson:** a constant fitted at one point is a curve evaluated once. It was not
merely imprecise away from that point — it was wrong in a way that changed which experiments ran.

### 5.2 A config can complete Vivado's flow and still not fit

`1x2000` was recorded `status: ok`. Vivado reported `place_design completed successfully`. It
does not fit:

```
post-synthesis : 20,126 LUTs  (96.76%)   <- fits, so placement proceeded
post-route     : 21,382 LUTs (102.80%)   <- physical optimization pushed it over
```

Placement starts from the post-synthesis netlist. Physical optimization then replicated logic
chasing timing and drove the routed design past the device.

**"Did it fit" has to be judged on measured routed area and timing, never on tool exit status.**
`dse/report.py` now does exactly that; its previous caveat keyed off `status == 'synth-failed'`
and would have gone on asserting that no edge had been found while the data plainly showed one.

### 5.3 Linear thermometer encoding is not representable in Q3.12

`emit_encoder.py` refused to build both linear configs. The reason is a real property of the
encoding, not a bug:

| encoding | threshold range | Q3.12 (±8) |
|---|---|---|
| distributive | [−4.545, 4.341] | fits |
| gaussian | [−2.578, 2.578] | fits |
| **linear** | [−6.134, **8.906**] | **23 of 3200 overflow** |

`Thermometer` spaces thresholds evenly from the data's min to max, so it reaches the outliers;
quantile and gaussian spacing stay in the bulk. Q3.12 was chosen in Phase 1 by measuring what
the *distributive* encoder needed — **the right fixed-point format depends on the encoding**,
and that does not generalize.

**Q4.11 would represent it at identical area** (still a 16-bit word, so identical comparator
cost) at a price of one fractional bit. Not pursued: threading `word_bits`/`frac_bits` through
five consumers is half a day's work, and the encoding axis shows a 0.12 pp spread — below the
noise floor — so the two points would not change any conclusion. Recorded as `gate1-failed`
with the reason rather than quietly dropped.

### 5.4 Bugs found in code that had never run

Several Phase 2 code paths existed for the sweep but had never executed. Auditing them
specifically found four defects, **each of which would have produced plausible-looking wrong
numbers rather than a visible failure**:

- **LUT tables were emitted shifted at n<3.** `np.packbits` pads a partial byte on the low side,
  so a 4-entry table emitted as `0x70` instead of `0x07` — every entry at the wrong address.
  n≥3 was unaffected, which is why n=6 never showed it. Worse: `dse-plan` §3 *predicts* n=2
  fails from routing congestion, so a Gate 1 failure here would have looked exactly like the
  expected architectural finding.
- **The `tau` schedule was interpolated linearly** where the paper's values are a power law in
  width, running 10–19% hot at every interpolated rung — while hitting the anchors exactly. That
  would have put a kink in the accuracy-vs-width curve that was an artifact of the schedule.
  Caught before the sweep; training was restarted.
- **The clock constraint reached Vivado — but Fmax could not prove it.** Fmax was 161.0 MHz at
  both 8 ns and 10 ns, correctly, since it reflects the same critical path. Only WNS separates
  a working `period` argument from a dropped one (+3.790 → +1.790, exactly the 2 ns removed).
  Had it been dropped, the clock axis would have read as "target clock has no effect."
- **Cross-session resume silently restarted from zero**, because `/kaggle/working` is fresh per
  version. A 34-config run that cannot fit one session would have repeated work indefinitely.

### 5.5 The emitter's self-check cannot catch an emitter bug

When the n=2 packing bug was deliberately reinstated to prove the test could fail, Gate 1
reported **958 mismatches of 1504** — and `emit_core.py`'s own read-back check reported
**20/20 nodes match**.

It parses its emitted Verilog and compares tables against `table_to_hex(...)`, *the same function
that had the bug*. The check is circular: it proves the file says what the emitter meant, never
that the emitter meant the right thing. **Only Gate 1, with an independent golden model, catches
this class of error** — which is why Gate 1 is non-negotiable rather than belt-and-braces.

### 5.6 The encoder narrowing result was fitted and tested on the same data

Phase 1 recorded per-feature comparator narrowing as −17.1% and "bit-exact (0 differences across
202 comparators × 1000 samples)." The area saving is real; the bit-exactness was not established.

`min_frac_bits()` searches for the fewest bits reproducing every comparison *for the samples
given* — and was given the 1000-sample vectors, then validated against those same 1000 samples.
Re-derived against the full 166k set, **8 of 15 features had been narrowed too far**. Feature 4
needs 12 fractional bits; the fit said 6.

The safe saving is roughly **−13%**, and it must be re-derived per config against the full test
set. Not adopting it was right — for a better reason than the one recorded.

---

## 6. Reproducing this

```bat
:: 0. prove the machine reproduces Phase 1 first -- 22/22, or the numbers are not comparable
.venv\Scripts\python.exe scripts\verify_phase1.py --with-board

:: 1. the grid and its budget
.venv\Scripts\python.exe dse\grid.py
.venv\Scripts\python.exe dse\grid.py --json build\dse\train_grid.json

:: 2. train (Kaggle, GPU) -- training/train_grid_kaggle.ipynb
::    ONLY_SLUGS / ONLY_Z / ONLY_ENCODING split the grid across sessions or accounts.
::    Download <slug>_checkpoint.pt AND <slug>_testvectors.npz into training/artifacts/sweeps/

:: 3. sanity-check what resolved before spending Vivado time
.venv\Scripts\python.exe dse\run.py --list

:: 4. the sweep: Gate 1 + place-and-route per config, resumable
.venv\Scripts\python.exe dse\run.py --all --impl

:: 5. results
.venv\Scripts\python.exe dse\report.py
.venv\Scripts\python.exe dse\report.py --snapshot     :: -> docs/results/, commit this
.venv\Scripts\python.exe dse\plot.py --snapshot
```

**Checkpoints do not travel with the repo** — ~933 MB across the grid, and one file exceeds
GitHub's 100 MB limit. `docs/results/` carries the measurements (232 KB) instead; §5 of
`docs/phase1-report.md` covers toolchain setup.

`dse/run.py` verifies each checkpoint's `n`, `z`, encoding, layers **and tau** against the grid
before building. That guard exists because the tau fix invalidated a set of checkpoints whose
filenames were identical to their replacements, and `tau` never reaches the hardware — a
wrong-vintage checkpoint produces valid RTL and plausible area with only its accuracy quietly
belonging to a different model.

---

## 7. What Phase 2 deliberately did not do

- **No board runs.** Phase 2 measures area and timing from Vivado reports; Gate 1b on silicon was
  Phase 1's exit condition and is not required per sweep point. Any config could be taken to
  hardware with `scripts/build_bitstream.py --rtl-dir build/configs/<name>/rtl`.
- **Clock variants on one rung only.** Pipeline depth was extended to four rungs (§4.6), but the
  8 ns / 12 ns clock targets were only tried at `1x360`. "What does asking for a different clock
  do" does not obviously need repeating per width, but that is an assumption rather than a
  measurement.
- **Only the width × z corner was explored.** The other axis pairs were ranked and rejected on
  measured penalties before spending GPU time: n=2 costs −1.25 pp against z=50's −0.24; n=4
  costs −0.38 pp for less saving than z=100's −0.10; multi-layer costs −0.75 pp; encoding varies
  by 0.12 pp, below the noise floor. **z × n is structurally pointless** — at z≤100 the encoder
  is already saturated, so cutting `n` barely shrinks it while still costing accuracy.
- **No multi-layer model at scale.** `2x1200` would carry a fifth pipeline stage and a smaller
  encoder than `1x2400`, which is the obvious candidate for making a 2400-node model meet
  100 MHz with headroom. It was ranked out on the single-layer accuracy advantage and never
  tried at that size.
- **No pipeline depths above 4.** `pipe_reg.ENABLE` is 0/1 and a single-layer model has exactly
  four register sites. If a config ever misses timing at 4 stages, the current RTL cannot rescue
  it — though a multi-layer model of similar size could, since each layer adds a stage.
- **Learnable Reduction not built.** ⚠️ **The reason recorded here was wrong.** It was declined
  on measuring 58 LUTs at `sm` — 3.6% of that design. But the reduction grows with layer width:
  at `1x2400 z=50` it is **4,450 LUTs, 34.9% of the design**, second only to the encoder. A LUT
  pyramid feeding a much narrower popcount projects to ~710 LUTs. The axis is reopened; see
  `docs/phase2-ledger.md`, 2026-08-10.
- **Configurable precision not implemented**, which is why the linear configs are unbuilt (§5.3).

---

## 8. Pointers

- `docs/phase2-ledger.md` — the dated log this was written from, including what was tried and rejected
- `docs/results/` — committed evidence: `frontier.png`, `area_split.png`, all measurements, the grid as trained
- `docs/dse-plan.md` — the plan, with superseded sections marked
- `docs/phase1-report.md` — what Phase 2 was built on, and toolchain setup
- `docs/reusable-generator.md` — scoping for packaging the generator as a tool
