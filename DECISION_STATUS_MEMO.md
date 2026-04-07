# Decision and Status Memo

## Status as of the completed schedule and multiseed studies
The active workflow remains model-based open-loop design of a plate-temperature reference trajectory for the inner PID, with the objective of obtaining approximately linear freezing-front progression.

## What changed
- The external knot-time schedule sensitivity study showed that the `N=3` versus `N=4` ranking is schedule-dependent.
- `uniform` is now the default external knot-time schedule for ongoing comparisons and workflow development.
- The uniform multiseed confirmation study showed that:
  - `N=3` is more stable across seeds
  - `N=4` shows upside but higher optimizer sensitivity
  - `N=4` does not yet demonstrate robust superiority
- Model order therefore remains unresolved under the default `uniform` schedule.

## Current status labels
- Historical reference workflow: `code_simulation/results/open_loop_final_workflow/locked_n3_seed29_init4_iter8/`
- Current default external schedule: `uniform`
- Pragmatic working default: `uniform + N=3`
- Unresolved challenger: `uniform + N=4`
- Scientific status: knot count unresolved

## Documentation consequence
Repository documentation should no longer present the locked `N=3` bundle as the current authoritative baseline in a strong scientific sense. It remains reproducible historical reference material. Current workflow docs should distinguish explicitly between historical references, pragmatic working defaults, and unresolved scientific conclusions.
