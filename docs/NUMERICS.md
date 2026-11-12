# Controlled numerical studies

The two controller tables were developed with the full physical model. The studies below retain those commands and their realized charging histories.

## Force ablation

Each case uses 644 ordered previous/current contexts. The four configurations retain full physics, disable flight Coulomb forces, disable wakes, or disable both. Electrode forces and charging induction remain enabled. All variants use the same simulation-defined reference phase. Experimental RMSE is evaluated on landed singletons; missing and merged targets are counted separately.

This measures sensitivity of the developed designs to the omitted terms. Independently recalibrated or reoptimized alternatives would require a separate comparison.

## Integration

Euler and RK4 share the force evaluator and grid-boundary event policy. Four predeclared contexts per case are evaluated at 1, 0.5 and 0.25 µs; RK4 at 0.125 µs provides the finest numerical reference. Release times remain fixed. Fourteen axial planes support trajectory comparisons independently of native endpoint overshoot. Interpolation eligibility and scored sample counts are recorded.

An analytic uniform-field/mass-loss case supplies an independent numerical check. Classical RK4 advances the continuous state at four stages; the discrete events retain their own accuracy limits.

## Timing

CPU and GPU runs use matched input hashes, batch sizes and precision. Records separate compilation, first execution, synchronized warm execution, transfer and output size. Two repetitions describe the measured workstation workload. A fully specified 32-particle synthetic case supports independent reproduction. Archived dense-recording results retain their original executable and output contracts.

## Data and provenance

Study manifests identify source code, inputs and outputs by hash. Experimental coordinates are the corresponding dot-position means from ten repeated A–Z prints per raster. Individual replicates, variances, raw images and calibration records are outside the supplied dataset; see [DATA.md](DATA.md).

Parameter inventories record geometry, charging calibration, controller settings and numerical configuration. The original CFD mesh and complete field-generation boundary conditions are outside the supplied material. The repository includes the imported field and application inputs; the manuscript source ZIP includes the article, figures and aggregate tables.
