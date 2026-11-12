# Experimental data

Each raster was used to print the same A–Z message ten times. The prints were analyzed from images, and corresponding dot positions were averaged across the ten measurements. The supplied data contain 1,725 mean dot positions per raster, in millimeters.

The low-speed case operates at 50 m/min and the high-speed case at 100 m/min. Both experimental comparisons use 75 kHz and 60 requested horizontal DPI. Dot identities are matched by message line, stroke column and logical landing position.

Measured coordinates are in `inputs/cases/{low,high}/measured_coordinates.csv`; matching simulations are in `simulated_coordinates.csv`. The current paired coordinates and metrics used for the figures are selected by `inputs/experimental_assessment.json` and stored in `results/experimental/`. The corrected high-speed file supersedes the earlier measurement version; the low-speed coordinates and all simulation predictions are unchanged. Superseded measurements are retained under `inputs/cases/high/history/`, and the earlier assessment remains in `results/reference/`. Reconstructed experimental prints show mean positions, with nominal deposited-dot diameters for visual context.

Reported coordinate RMSE compares the simulation with the ten-print mean. Additional laboratory records, including raw images, individual replicate measurements and acquisition details, remain confidential and are not distributed. Per-dot variances are outside the released dataset. The mean arrays therefore do not determine repeatability, confidence intervals or single-print error.

Only the final coordinate version is used in the reported comparisons. Earlier acquisition descriptions are superseded by this sampling protocol; coordinate revisions are not additional replicate prints. Input hashes and revision relationships retain the data lineage.
