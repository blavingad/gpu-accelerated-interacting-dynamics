# Reproduction guide

## Coordinate assessment and figures

`workflows/assess_prints.py` recomputes paired-coordinate errors for the assessment selected in `inputs/experimental_assessment.json`. Revisions are archived and rescored with `update_measurements.py`; cached physical trajectories are retained. `reproduce_figures.py` regenerates reference result figures, and `draw_methodology.py` renders the framework schematic. These operations use the supplied data without running an optimization.

`build_manuscript.py` verifies frozen study manifests, regenerates the controlled-study figures and builds the manuscript. It does not alter Git history. Tectonic is required for PDF compilation and may be selected through `CIJ_TECTONIC`.

## Numerical verification

Run `python -m unittest discover -s tests`, followed by `verify_uniform_field.py --device cpu` and `verify_kernel.py`. The analytic case checks an event-free uniform-field/mass-loss limit. Preserved solver fixtures establish Euler compatibility, endpoint/lineage/gutter parity and batched versus scalar agreement.

The controlled integration study uses identical realized charges and native release times for Euler and RK4. Refinement changes the integration step while retaining release timing and event policy. RK4 at 0.125 µs is a numerical reference, not an exact solution of the hybrid system.

## Simulation and optimization

Native case assessments and new optimization outputs are written under `results/generated/`; reference inputs remain unchanged. Full-pool assessment evaluates 644 ordered previous/current contexts per case. Detailed force and trajectory recording is optional.

The structural search retains permutation screening, coupled assessment, finite-difference sensitivities, bounded least squares, Broyden updates and integer replay. Shared charge fitting and fixed-command fitting are separate workflows. A charge-optimized result requires controller realization and waveform replay before deployment.

The archived structural campaign uses 76.8 kHz. The two physical-assessment cases use 75 kHz and retain their recorded droplet properties. These campaigns are not a controlled comparison isolating structure alone. New optimization traces may differ across hardware and precision.

## Inputs and evidence

The precomputed field, case configurations, controller tables and final coordinate means are provided in `inputs/`. Imported-field replay is supported; reconstructing the original CFD mesh and field solve is outside the supplied material. Experimental sampling is described in [DATA.md](DATA.md).

Frozen studies record source/input hashes and numerical settings. Historical snapshots retain their original executable hashes; current sources and archived reference implementations are distinguished. Do not edit a frozen study in place. New physical or numerical settings require a separate, versioned study.

The manuscript source ZIP contains the article, figures and aggregate tables. Simulation inputs are distributed in the repository, separately from the arXiv source package.

Historical paths recorded in frozen manifests are mapped to their current locations in `docs/provenance/paths.json`. The numerical files and their hashes are unchanged.

To replace the high-speed measurements and rescore saved predictions:

```bash
python workflows/update_measurements.py --high measurements.csv --output results/measurement_review
python workflows/reproduce_figures.py
python workflows/build_manuscript.py
```

Choose an empty output directory. Earlier coordinates and assessments are retained.
