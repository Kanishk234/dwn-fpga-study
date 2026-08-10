# Documentation index

Start with a **report** if you want results, a **ledger** if you want the reasoning and the
wrong turns, and the **brief** if you want the plan the whole thing follows.

## Reports — what happened, written up

| | |
|---|---|
| [`phase1-report.md`](./phase1-report.md) | Getting one model onto a Basys 3: what was built, the five things that cost time, and copy-pasteable steps to reproduce every number on another machine. |
| [`phase2-report.md`](./phase2-report.md) | The design-space exploration: 46 configurations, the full results table, the Pareto frontier, and the six things that broke. |
| [`results/`](./results/) | The committed evidence — every measurement, both figures, and the grid as trained. 232 KB. |

## Ledgers — the dated working logs

These are the raw record the reports were written from. They keep **what was tried and
rejected**, and corrections stay visible rather than being edited away — so a wrong turn is
still readable.

| | |
|---|---|
| [`phase1-ledger.md`](./phase1-ledger.md) | Phase 1, day by day |
| [`phase2-ledger.md`](./phase2-ledger.md) | Phase 2, day by day |

## Plans and specifications

| | |
|---|---|
| [`project-brief.md`](./project-brief.md) | The full technical plan: resource budgets (§6), the two studies (§10), phase breakdown (§11), risks (§12) |
| [`dse-plan.md`](./dse-plan.md) | What Phase 2 sweeps and why. **Partly superseded** — §5's encoder assumption and §7's open questions are marked where measurement overtook them |
| [`phase2-handoff.md`](./phase2-handoff.md) | Moving to a new machine, and the restructure Phase 2 carried out |
| [`phase3-plan.md`](./phase3-plan.md) | The controlled comparison: what to build, what to measure, and the two comparability traps |

## Reference — facts established once, relied on everywhere

| | |
|---|---|
| [`checkpoint-format.md`](./checkpoint-format.md) | What the exporter reads, verified against the pinned upstream commit. Includes the traps — address bit order, and a dummy mapping that looks exactly like a real one |
| [`paper-configs.md`](./paper-configs.md) | The paper's JSC configurations (Table 14 / Table 2) and what they corrected |
| [`probe-results.md`](./probe-results.md) | Phase 1a: does `TABLE[addr]` map to a single LUT6? The evidence the whole area model rests on |

## Scoping — considered, not started

| | |
|---|---|
| [`reusable-generator.md`](./reusable-generator.md) | What it would take to package the RTL generator as a tool others could use, in the way hls4ml is used. Not part of the project; recorded while the shape of the work is clear |
