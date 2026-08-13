# Documentation index

Start with a **report** if you want results, a **ledger** if you want the reasoning and the
wrong turns, and the **brief** if you want the plan the whole thing follows.

**Files at this level are the JSC study** — the original dataset, and everything `docs/jsc-report.md`
describes. The second dataset lives in its own directory: [`mnist/`](./mnist/), indexed
[below](#the-mnist-study). The asymmetry is deliberate rather than tidy: `docs/jsc-report.md`, the root
`README.md` and the `jsc-complete` tag all reference these paths, and breaking a published
artifact to gain symmetry is a bad trade. Dimensions and per-dataset settings live in
`datasets/`, never in the shared code.

⚠️ **The two studies are not comparable on shared axes** and are deliberately never merged into
one table or one figure: their accuracy scales differ by ~20 points, their encoder economics run
opposite, and their figures are pinned to different tags (`jsc-complete` and `mnist-complete`).

## Reports — what happened, written up

| | |
|---|---|
| [`phase1-report.md`](./phase1-report.md) | Getting one model onto a Basys 3: what was built, the five things that cost time, and copy-pasteable steps to reproduce every number on another machine. |
| [`phase2-report.md`](./phase2-report.md) | The design-space exploration: 52 measured configurations, the full results table, the Pareto frontier, and the six things that broke. |
| [`phase3-report.md`](./phase3-report.md) | The controlled comparison: DWN against conifer and hls4ml on identical silicon, plus two defects in how this field compares results. |
| [`results/`](./results/) | Phase 2 evidence — all 54 records (52 measured, 2 unbuildable), both figures, and the grid as trained. 308 KB. |
| [`results-cc/`](./results-cc/) | Phase 3 evidence — 14 conifer and 6 hls4ml configurations measured through the same flow, plus both comparison figures. |

## Ledgers — the dated working logs

These are the raw record the reports were written from. They keep **what was tried and
rejected**, and corrections stay visible rather than being edited away — so a wrong turn is
still readable.

| | |
|---|---|
| [`phase1-ledger.md`](./phase1-ledger.md) | Phase 1, day by day |
| [`phase2-ledger.md`](./phase2-ledger.md) | Phase 2, day by day |
| [`phase3-ledger.md`](./phase3-ledger.md) | Phase 3, day by day — both halves, with the machine that produced each row noted |

## Plans and specifications

| | |
|---|---|
| [`project-brief.md`](./project-brief.md) | The full technical plan: resource budgets (§6), the two studies (§10), phase breakdown (§11), risks (§12) |
| [`dse-plan.md`](./dse-plan.md) | What Phase 2 sweeps and why. **Partly superseded** — §5's encoder assumption and §7's open questions are marked where measurement overtook them |
| [`phase2-handoff.md`](./phase2-handoff.md) | Moving to a new machine, and the restructure Phase 2 carried out |
| [`phase3-plan.md`](./phase3-plan.md) | The controlled comparison: what to build, what to measure, and the two comparability traps |
| [`phase3-handoff.md`](./phase3-handoff.md) | Running Phase 3 on another machine: acceptance test, toolchain, and the rules that keep the comparison controlled |

## Reference — facts established once, relied on everywhere

| | |
|---|---|
| [`datapath.md`](./datapath.md) | What each stage of the design actually does — encoder, LUT layer, popcount, argmax — what each costs, and how much of it generalizes beyond JSC |
| [`checkpoint-format.md`](./checkpoint-format.md) | What the exporter reads, verified against the pinned upstream commit. Includes the traps — address bit order, and a dummy mapping that looks exactly like a real one |
| [`paper-configs.md`](./paper-configs.md) | The paper's JSC configurations (Table 14 / Table 2) and what they corrected |
| [`probe-results.md`](./probe-results.md) | Phase 1a: does `TABLE[addr]` map to a single LUT6? The evidence the whole area model rests on |

## The MNIST study

The same flow on a second, deliberately different dataset — 784 features against 16, ten classes
against five. **The port is the result**: the exporter, generator and testbenches became
dataset-agnostic, and both datasets now reproduce from the same code.

| | |
|---|---|
| [`mnist/report.md`](./mnist/report.md) | **Start here** — the standalone MNIST study, in the same form as `docs/jsc-report.md` |
| [`mnist/phase1-report.md`](./mnist/phase1-report.md) | Bring-up: one bug pattern found seven times, and what generalising actually required |
| [`mnist/phase2-report.md`](./mnist/phase2-report.md) | The sweep: 25 configurations, and two Phase 1 predictions retracted |
| [`mnist/phase3-report.md`](./mnist/phase3-report.md) | The comparison: DWN against its own weightless family for the first time |
| [`results-mnist/`](./results-mnist/) | Phase 2 evidence — all 25 measured records, both figures, the grid as trained |
| [`results-cc-mnist/`](./results-cc-mnist/) | Phase 3 evidence — the conifer measurements and the comparison figure |
| [`mnist/plan.md`](./mnist/plan.md) | The ground rules the port was held to, including the contract in §1.5 |

Ledgers, same role as above — the dated record with the wrong turns left visible:
[`phase1-`](./mnist/phase1-ledger.md) · [`phase2-`](./mnist/phase2-ledger.md) ·
[`phase3-`](./mnist/phase3-ledger.md) · [`reduction-ledger.md`](./mnist/reduction-ledger.md),
which measured the 0.24 pp noise floor that withdrew three claims — including the study's own
headline.

## Scoping — considered, not started

| | |
|---|---|
| [`reusable-generator.md`](./reusable-generator.md) | What it would take to package the RTL generator as a tool others could use, in the way hls4ml is used. Not part of the project; recorded while the shape of the work is clear |
