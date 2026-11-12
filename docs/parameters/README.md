# Model settings

The JSON files record geometry bounds, charging parameters, physical configuration and controller settings. `input_inventory.json` lists input hashes; `environment.json` records the numerical runtime.

- Both assessment cases use 75 kHz, with their recorded droplet diameter, density and velocity.
- The amplifier time constant is 0.5 µs and breakup is sampled at mid-cycle. The 820 pF electronics load is separate from effective droplet capacitance.
- Guard and consecutive-print compensation commands are disabled. Neighbor-induced charging remains enabled.
- The reference emits a preceding and current stroke, with blank older history and no following stroke.
- Particle state is float32; waveform preparation uses NumPy float64. Release and event tests use the time grid.
- Euler and RK4 share force and event rules. The integration study fixes release times while refining the integration step.
- Positive-air flow is zero in the application studies.
- Permanent merging follows the target-eligibility mask; neutral guards and preceding-stroke particles are not generally merge eligible.
- Experimental coordinates are ten-print means. Further laboratory records remain confidential. The current dataset is selected by `inputs/experimental_assessment.json`.
- The supplied field supports replay. Its original CFD mesh and full field-generation archive are outside this repository.

Run `python workflows/describe_inputs.py` to regenerate parameter and environment records.
