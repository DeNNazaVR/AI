# Implementation Notes

Very short list of issues to keep updating during review.

- Don't push .venv, create a requirement.txt instead
- For now, make the ranodm sampling deteministic: "repeating the experiments several time gives the same explanation".



| Location | Issue | Why wrong | Fix |
| --- | --- | --- | --- |
| `interval_pattern.py:44-61` | `meet` accepts patterns with different lengths. | Interval patterns should have the same feature dimension; appending extra segments hides data/model bugs. | Require equal lengths and raise `ValueError` otherwise. |
| `interval_pattern.py:63-69` | `covers(p, o)` name is ambiguous. | It only checks whether point/object `o` lies inside pattern `p`; it is not the Hasse cover relation. | Rename later to `contains_object` or `pattern_contains_object`. The order relation is `is_more_general`; graph covers are built in `explanation.py:318-339`. |
| `interval_pattern.py:78-99`, `explanation.py:276` | `closure(p, local_samples)` may exclude the explained object. | In the pipeline, `local_samples` does not include `o`, so closure can shrink a candidate until it no longer describes `o`. | Close over `[o] + local_samples`, or discard any closed pattern where `not covers(p, o)`. |
| `interval_pattern.py`, `local_neighborhood.py`, `explanation.py` | Interval helpers/classes are repeated. | Fixes in one copy will not update the others; behavior can silently diverge. | Keep one implementation, import it everywhere else. |
| `explanation.py:425` | `need_purity = 0.2` is too low. | For binary classification, 0.2 can accept patterns mostly disagreeing with `f(o)`. | Use a high experimental threshold, e.g. `0.8`/`0.9`, or report purity trade-offs explicitly. |
| `explanation.py:448-453` | Final choice maximizes support before purity. | Support matters, but purity is at least as important for an explanation of one prediction. | Rank by purity first, use a combined score, or expose Pareto-optimal candidates. |
| `explanation.py:428-455` | No formal requirement that final pattern covers `o`. | A selected explanation may not actually describe the explained object. | Add `covers(pattern, o)` to candidate filtering/final selection. |

