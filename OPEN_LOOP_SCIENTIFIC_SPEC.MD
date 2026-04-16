## 9. Current uncertainty that is NOT scientifically closed

The following are not yet scientifically closed:

- the scientifically adequate trajectory parameterization,
- the adequate external time parameterization,
- whether fixed time parameterizations are an acceptable approximation,
- the sensitivity of the optimization outcome to front definition and dT_mushy.

No document or result should anchor the active scientific discussion on any preselected knot count.

## 10. Immediate technical priority

The immediate priority is NOT to run more large BO studies.

The immediate priority is:

1. clean documentation and align code defaults with the actual intended workflow,
2. fix the operational front definition,
3. run a narrow sensitivity study on front extraction and dT_mushy,
4. only then run a narrow, pre-registered BO comparison of trajectory parameterizations.

## 11. Near-term optimization study scope

The first clean BO study should:

- use a single fixed schedule family,
- use the same objective function and admissibility rules for all cases,
- compare a small number of candidate trajectory parameterizations only,
- use fixed seeds and fixed budget,
- evaluate not only best objective, but also stability, feasibility, and executability.

The first clean BO study should not assume in advance that the key scientific question is a specific knot-count choice.
