# Numerical evidence

The force-ablation, integration and benchmark directories retain the frozen numerical studies and their original source/input hashes. Cached particle predictions remain the references for later measurement assessments.

The experimental scores embedded in the original ablation directory use the earlier experimental coordinate version. Current experimental metrics and rescored ablation comparisons are selected by `inputs/experimental_assessment.json`. The corrected high-speed measurements are assessed in `results/experimental/`, using the same frozen trajectories and reference phase. Earlier measurements are retained in `inputs/cases/high/history/`.

`publication.json` identifies the numerical freeze and experimental assessment used for the current manuscript. A measurement revision updates experimental comparisons, not the frozen particle predictions or integration/benchmark results.
