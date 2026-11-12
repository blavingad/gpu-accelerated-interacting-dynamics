# Public synthetic uniform-field case

This case uses no industrial geometry, font, raster sequence or DAC table. It verifies the retained mass-loss/force/time-integration equations against an analytic event-free limit. It is numerical verification, not experimental validation.

Uniform field: E=(-300000,0,0) V/m. Charge: -1 pC. Diameter: 132 µm. Density: 905 kg/m³. Axial velocity: 24 m/s. Drag, gravity, wake and Coulomb interactions are disabled. The implemented area-scaled mass loss is retained.

With k the initial-mass loss coefficient per distance, alpha=k*vz/(3*m0), the continuous solution is m(t)=m0*(1-alpha*t)^3 and X(t)=q*Ex*t²/[2*m0*(1-alpha*t)]. Reported time is the actual number of discrete steps times dt. Single-precision integration and field interpolation are retained. No landing-time interpolation is needed for this event-free verification window.

Run `python run_public_uniform_field_benchmark.py --output output --device cpu`. Install compatible NumPy, pandas, JAX and jaxlib. Accelerator installation depends on the target platform. Exact package versions used internally are in environment.json.
