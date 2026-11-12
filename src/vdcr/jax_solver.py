"""JAX/XLA integration kernel for CIJ droplet dynamics."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .config import SimulationConfig
from .electric_field import load_electric_field_grid, trilinear_field


def sphere_drag_coefficient(reynolds_number):
    """Schiller–Naumann sphere drag, transitioning to the Newton regime."""
    reynolds_number = jnp.asarray(reynolds_number, dtype=jnp.float32)
    safe_reynolds = jnp.maximum(reynolds_number, jnp.float32(1e-6))
    finite_re_drag = (24.0 / safe_reynolds) * (
        1.0 + 0.15 * safe_reynolds**0.687
    )
    return jnp.where(reynolds_number < 1000.0, finite_re_drag, 0.44)


def _correlated_air_coefficients(config: SimulationConfig, steps: int):
    """Generate stationary OU modal amplitudes and spatial Fourier phases."""
    modes = int(config.positive_air_random_modes)
    rng = np.random.default_rng(int(config.positive_air_seed))
    rho = float(np.exp(-config.dt / config.positive_air_correlation_time_s))
    innovation = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    coefficients = np.empty((steps, 3, modes), dtype=np.float32)
    coefficients[0] = rng.standard_normal((3, modes)).astype(np.float32)
    forcing = rng.standard_normal((max(0, steps - 1), 3, modes)).astype(np.float32)
    for step in range(1, steps):
        coefficients[step] = rho * coefficients[step - 1] + innovation * forcing[step - 1]
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(3, modes)).astype(np.float32)
    # Geometrically separated wavelengths prevent a single artificial period.
    wavelength_ratios = np.geomspace(1.0, 4.0, modes, dtype=np.float32)
    wavenumbers = (
        2.0 * np.pi
        / (config.positive_air_correlation_length_m * wavelength_ratios)
    ).astype(np.float32)
    return coefficients, phases, wavenumbers


def resolve_device(requested=None):
    if isinstance(requested, jax.Device):
        return requested
    if requested is None or requested == "auto":
        return jax.devices()[0]
    if requested in {"cpu", "gpu", "tpu"}:
        devices = jax.devices(requested)
        if devices:
            return devices[0]
        raise RuntimeError(f"No JAX {requested.upper()} device is available")
    for device in jax.devices():
        if str(device) == requested:
            return device
    raise ValueError(f"Unknown device {requested!r}; available: {jax.devices()}")


def prepare_integrator(config: SimulationConfig, requested_device=None, *, output_mode="full", gutter_z_mm=43.770, method="euler", release_times_s=None, sample_stations_mm=()):
    """Prepare one reusable executable family with dynamic charges and target masks.

    Geometry, physical constants, capacity and time grid are static. Force and event
    equations are shared by full recording and lean endpoint modes. Charges are
    dynamic in coulombs at the compiled-kernel interface; public run() takes pC.
    """
    from copy import deepcopy
    config = deepcopy(config)
    config.validate()
    if output_mode not in {"full", "endpoints"}:
        raise ValueError("output_mode must be full or endpoints")
    if output_mode == "endpoints" and config.openfoam_field_path is None:
        raise ValueError("Endpoint diagnostics currently require coupled +Z flight")
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be euler or rk4")
    stations = np.asarray(sample_stations_mm, dtype=float)
    if stations.ndim != 1 or not np.isfinite(stations).all():
        raise ValueError("sample_stations_mm must be a finite one-dimensional sequence")
    device = resolve_device(requested_device)
    # Device selection must precede even scalar/initial-state allocation;
    # otherwise a visible GPU may receive arrays before an explicit CPU request.
    jax.config.update("jax_default_device", device)
    n = config.particles_y_dim
    steps = int(np.ceil(config.max_simulation_time / config.dt))
    ids = jnp.arange(n)
    coupled = config.openfoam_field_path is not None
    base_position = jnp.asarray(config.initial_position_mm, jnp.float32) * 1e-3
    pos = jnp.broadcast_to(base_position, (n, 3)).copy()
    if not config.sequential_injection:
        pos = pos.at[:, 1].add(-ids * config.y_spacing)
    vel = jnp.zeros((n, 3), jnp.float32)
    vel = vel.at[:, 0].set(config.initial_x_velocity)
    vel = vel.at[:, 1].set(config.initial_y_velocity)
    vel = vel.at[:, 2].set(config.initial_z_velocity)
    charge = jnp.asarray(config.droplet_charges_coulombs, jnp.float32)
    # A file-driven raster may contain electrically active compensation pulses.
    # They participate in every force but are not physical print targets and
    # therefore cannot form a permanent printed-droplet coalescence.
    coalescence_eligible = jnp.asarray(
        config.intended_print_mask
        if config.intended_print_mask is not None
        else np.abs(config.droplet_charges_pC) > 1e-12,
        dtype=bool,
    )
    initial_pos = pos
    active = jnp.ones(n, bool)
    zeros_b = jnp.zeros(n, bool)
    landing_pos = jnp.zeros((n, 3), jnp.float32)
    mass0 = jnp.full(n, config.initial_particle_mass, jnp.float32)
    radius0 = jnp.full(n, config.initial_droplet_radius, jnp.float32)
    acc0 = jnp.zeros((n, 3), jnp.float32)
    eye = jnp.eye(n, dtype=bool)
    dt = jnp.float32(config.dt)
    eps = jnp.float32(1e-12)
    body = jnp.asarray(config.body_force_vector, jnp.float32)
    substrate = jnp.asarray(config.substrate_velocity_vector, jnp.float32)
    positive_air_mean = jnp.asarray(config.positive_air_velocity_vector, jnp.float32)
    air_coefficients_np, air_phases_np, air_wavenumbers_np = _correlated_air_coefficients(config, steps)
    air_coefficients = jnp.asarray(air_coefficients_np, jnp.float32)
    air_phases = jnp.asarray(air_phases_np, jnp.float32)
    air_wavenumbers = jnp.asarray(air_wavenumbers_np, jnp.float32)
    unsteady_scale = jnp.float32(config.positive_air_unsteady_scale)
    if coupled:
        air_rms = jnp.asarray(
            [config.positive_air_crossflow_rms_mps, config.positive_air_crossflow_rms_mps,
             config.positive_air_axial_rms_mps], jnp.float32
        ) * unsteady_scale
    else:
        air_rms = jnp.asarray(
            [config.positive_air_crossflow_rms_mps, config.positive_air_axial_rms_mps,
             config.positive_air_crossflow_rms_mps], jnp.float32
        ) * unsteady_scale
    vortex_strength = jnp.float32(config.positive_air_vortex_strength_mps) * unsteady_scale
    modal_normalization = jnp.sqrt(jnp.float32(2.0 / config.positive_air_random_modes))
    release_steps = jnp.rint(ids / (config.nozzle_frequency_hz * config.dt)).astype(jnp.int32)
    if release_times_s is not None:
        fixed_releases = np.asarray(release_times_s, dtype=float)
        if fixed_releases.shape != (n,) or not np.isfinite(fixed_releases).all() or np.any(fixed_releases < 0):
            raise ValueError("release_times_s must contain one finite nonnegative release per particle")
        release_steps = jnp.asarray(np.rint(fixed_releases / config.dt), jnp.int32)
    station_planes = jnp.asarray(stations * 1e-3, jnp.float32)

    if coupled:
        grid = load_electric_field_grid(config.openfoam_field_path)
        field_values = jnp.asarray(grid.values, jnp.float32)
        field_origin = jnp.asarray(grid.origin, jnp.float32)
        field_spacing = jnp.asarray(grid.spacing, jnp.float32)
        field_upper = jnp.asarray(grid.upper_bound, jnp.float32)
    else:
        field_values = jnp.zeros((2, 2, 2, 3), jnp.float32)
        field_origin = jnp.zeros(3, jnp.float32)
        field_spacing = jnp.ones(3, jnp.float32)
        field_upper = jnp.ones(3, jnp.float32)

    def forces(pos, vel, mass, radius, charge, active_now, step_index, stage_fraction=0.):
        charged = jnp.abs(charge) > jnp.float32(1e-15)
        flight_axis = 2 if coupled else 1
        area = jnp.pi * radius**2
        # Nearest upstream wake source. Coupled flight is +Z; the legacy case
        # flies -Y. Matrix row i is the affected downstream droplet and column
        # j is its candidate upstream wake source.
        flight_sign = 1.0 if coupled else -1.0
        separation = pos[:, None, :] - pos[None, :, :]
        # Wake lies behind source j: axial distance is source progress minus
        # target progress, positive when target i trails source j.
        axial_separation = -flight_sign * separation[:, :, flight_axis]
        radial_squared = jnp.sum(separation**2, axis=2) - separation[:, :, flight_axis] ** 2
        radial = jnp.sqrt(jnp.maximum(radial_squared, 0.0))
        # Columns are wake sources. Use source radius and speed, not the
        # affected target's values. The wake becomes valid one source diameter
        # downstream and expands conically thereafter.
        source_radius = radius[None, :]
        source_speed = jnp.abs(vel[None, :, flight_axis])
        wake_radius = source_radius + axial_separation * jnp.tan(config.wake_spread_angle_rad)
        min_wake_length = config.wake_min_axial_length_factor * source_radius
        wake_mask = (
            active_now[:, None] & active_now[None, :] & ~eye
            & (axial_separation >= min_wake_length)
            & (axial_separation < config.wake_max_axial_length)
            & (radial < wake_radius)
        )

        # Conservative momentum-deficit surrogate. Sphere drag supplies the
        # momentum removed from the air; distributing it over the expanding
        # wake area gives a centreline deficit proportional to
        # Cd * (source_radius / wake_radius)^2. The calibration factor permits
        # uncertainty studies without returning to an arbitrary fraction of
        # particle speed and an independently chosen decay exponent.
        source_reynolds = (
            config.air_density * source_speed * (2.0 * source_radius)
            / config.air_dynamic_viscosity_pa_s
        )
        source_cd = sphere_drag_coefficient(source_reynolds)
        centre_fraction = (
            config.wake_initial_velocity_deficit_factor * 0.5 * source_cd
            * (source_radius / jnp.maximum(wake_radius, eps)) ** 2
        )
        centre_fraction = jnp.minimum(
            centre_fraction, config.wake_max_velocity_deficit_fraction
        )
        wake_base = source_speed * centre_fraction
        ratio = jnp.maximum(0, radial / jnp.maximum(wake_radius, eps))
        radial_decay = jnp.exp(
            -config.wake_radial_gaussian_coefficient * ratio**2
        )
        strength = wake_mask * wake_base * radial_decay
        nearest = jnp.argmin(jnp.where(wake_mask, axial_separation, jnp.inf), axis=1)
        direction = vel / (jnp.linalg.norm(vel, axis=1, keepdims=True) + eps)
        deficit = direction[nearest] * strength[ids, nearest, None] * jnp.any(wake_mask, axis=1)[:, None]
        deficit_speed = jnp.linalg.norm(deficit, axis=1, keepdims=True)
        deficit_cap = config.wake_max_velocity_deficit_fraction * jnp.abs(vel[:, flight_axis:flight_axis + 1])
        deficit *= jnp.where(deficit_speed > deficit_cap + eps, deficit_cap / (deficit_speed + eps), 1)

        fg = mass[:, None] * body * active_now[:, None]
        y = pos[:, 1]
        tapered_gap = config.straight_plate_distance + jnp.abs(y - config.primary_deflection_end_y) * jnp.tan(config.secondary_plate_angle_rad)
        primary = (y <= config.deflection_plate_start_y) & (y >= config.primary_deflection_end_y)
        secondary = (y < config.primary_deflection_end_y) & (y >= config.secondary_deflection_end_y)
        gap = jnp.where(primary, config.straight_plate_distance, jnp.where(secondary, tapered_gap, jnp.inf))
        ex = config.plate_voltage_diff / gap
        analytic_field = jnp.stack((ex, jnp.zeros_like(ex), ex * config.electric_field_z_ratio), axis=1)
        field = trilinear_field(pos, field_values, field_origin, field_spacing) if coupled else analytic_field
        fe = charge[:, None] * field * active_now[:, None]

        rvec = pos[:, None, :] - pos[None, :, :]
        r2 = jnp.sum(rvec**2, axis=2)
        min_r2 = (radius[:, None] + radius[None, :])**2
        if config.coulomb_cutoff_distance_mm is None:
            within_coulomb_cutoff = jnp.ones_like(r2, dtype=bool)
        else:
            cutoff_m = jnp.asarray(config.coulomb_cutoff_distance_mm * 1e-3, dtype=jnp.float32)
            within_coulomb_cutoff = r2 <= cutoff_m * cutoff_m
        cmask = active_now[:, None] & active_now[None, :] & charged[:, None] & charged[None, :] & ~eye & within_coulomb_cutoff & (r2 > min_r2 * 1.01)
        inv_r3 = jnp.where(cmask, jnp.maximum(r2, eps**2)**-1.5, 0)
        fc_pair = config.coulomb_constant * (charge[:, None] * charge[None, :])[:, :, None] * rvec * inv_r3[:, :, None]
        fc = jnp.sum(fc_pair, axis=1)
        if not config.coulomb_interaction_enabled:
            fc_pair = jnp.zeros_like(fc_pair)
            fc = jnp.zeros_like(fc)

        # Base drag is referenced to still air. Wake and imposed positive-air
        # co-flow are retained as explicit additive corrections for diagnostics.
        relative0 = vel
        def drag(relative):
            speed = jnp.linalg.norm(relative, axis=1, keepdims=True)
            diameter = 2.0 * radius[:, None]
            reynolds = (
                config.air_density
                * speed
                * diameter
                / config.air_dynamic_viscosity_pa_s
            )
            drag_coefficient = sphere_drag_coefficient(reynolds)
            return -0.5 * config.air_density * relative * speed * area[:, None] * drag_coefficient * active_now[:, None]
        fd0 = drag(relative0)
        fd_wake = drag(relative0 - deficit)
        fw = fd_wake - fd0
        has_wake_source = jnp.any(wake_mask, axis=1)
        fw_pair = jnp.zeros((n, n, 3), jnp.float32).at[ids, nearest, :].set(fw)
        fw_pair = fw_pair * has_wake_source[:, None, None]

        # Synthetic, correlated carrier flow. OU modal amplitudes give a finite
        # correlation time; Fourier modes give a finite axial correlation
        # length. A rotating coherent mode represents repeatable recirculation.
        progress = flight_sign * pos[:, flight_axis]
        modal_shape = jnp.cos(
            progress[:, None, None] * air_wavenumbers[None, None, :]
            + air_phases[None, :, :]
        )
        stochastic_unit = modal_normalization * jnp.sum(
            modal_shape * air_coefficients[step_index][None, :, :], axis=2
        )
        stochastic_air = stochastic_unit * air_rms[None, :]
        vortex_phase = (
            jnp.float32(2.0 * jnp.pi * config.positive_air_vortex_frequency_hz)
            * ((step_index.astype(jnp.float32) + stage_fraction) if stage_fraction else step_index.astype(jnp.float32)) * dt
            + jnp.float32(2.0 * jnp.pi / config.positive_air_correlation_length_m)
            * progress
        )
        if coupled:
            coherent_air = vortex_strength * jnp.stack(
                (jnp.sin(vortex_phase), jnp.cos(vortex_phase), jnp.zeros_like(vortex_phase)),
                axis=1,
            )
        else:
            coherent_air = vortex_strength * jnp.stack(
                (jnp.sin(vortex_phase), jnp.zeros_like(vortex_phase), jnp.cos(vortex_phase)),
                axis=1,
            )
        air_velocity = positive_air_mean[None, :] + stochastic_air + coherent_air
        fd = drag(relative0 - deficit - air_velocity)
        fa = fd - fd_wake
        # Explicit decomposition: total drag = still-air drag + wake +
        # positive-air correction. This avoids counting any drag twice.
        fnet = fg + fd0 + fw + fa + fe + fc
        acc = jnp.where(active_now[:, None], fnet / mass[:, None], 0)
        return acc, fg, fd0, fd, fe, fc, fc_pair, fw, fw_pair, fa, air_velocity, fnet

    def one_step(carry, step_index, coalescence_eligible):
        old_carry = carry
        pos, vel, active, landed, discarded, landing_pos, mass_previous, radius_previous, acc, charge, lineage = carry
        running = jnp.any(active)
        released = (step_index >= release_steps) if config.sequential_injection else jnp.ones(n, bool)
        active_now = active & released
        active_before = active_now
        charged = jnp.abs(charge) > jnp.float32(1e-15)
        flight_axis = 2 if coupled else 1
        distance_increment = jnp.abs(vel[:, flight_axis]) * dt
        area_scale = (radius_previous / jnp.float32(config.initial_droplet_radius)) ** 2
        evaporated_mass = config.mass_loss_rate_per_meter * distance_increment * area_scale
        mass = jnp.where(active_now, jnp.maximum(mass_previous - evaporated_mass, 1e-20), mass_previous)
        radius = (mass / (((4.0 / 3.0) * jnp.pi) * config.initial_density)) ** (1.0 / 3.0)
        area = jnp.pi * radius**2

        if method == "euler":
            # Original mass-first, velocity-first update is the compatibility path.
            acc, fg, fd0, fd, fe, fc, fc_pair, fw, fw_pair, fa, air_velocity, fnet = forces(pos, vel, mass, radius, charge, active_now, step_index)
            force_mass, force_radius, force_acceleration = mass, radius, acc
            vel = jnp.where(active_now[:, None], vel + acc * dt, vel)
            pos = jnp.where(active_now[:, None], pos + vel * dt, pos)
        else:
            # Integrate position, velocity and mass jointly. Discrete releases,
            # contact pairing and landing tests occur only at the common grid boundaries.
            def rhs(x, v, m, fraction):
                m = jnp.maximum(m, 1e-20)
                r = (m / (((4. / 3.) * jnp.pi) * config.initial_density)) ** (1. / 3.)
                diagnostic = forces(x, v, m, r, charge, active_now, step_index, fraction)
                dm = -config.mass_loss_rate_per_meter * jnp.abs(v[:, flight_axis]) * (r / jnp.float32(config.initial_droplet_radius)) ** 2
                return jnp.where(active_now[:, None], v, 0), diagnostic[0], jnp.where(active_now, dm, 0), diagnostic
            x0, v0, m0 = pos, vel, mass_previous
            k1x, k1v, k1m, diagnostics = rhs(x0, v0, m0, 0.)
            k2x, k2v, k2m, _ = rhs(x0 + .5*dt*k1x, v0 + .5*dt*k1v, m0 + .5*dt*k1m, .5)
            k3x, k3v, k3m, _ = rhs(x0 + .5*dt*k2x, v0 + .5*dt*k2v, m0 + .5*dt*k2m, .5)
            k4x, k4v, k4m, _ = rhs(x0 + dt*k3x, v0 + dt*k3v, m0 + dt*k3m, 1.)
            pos = jnp.where(active_now[:, None], x0 + dt/6*(k1x + 2*k2x + 2*k3x + k4x), x0)
            vel = jnp.where(active_now[:, None], v0 + dt/6*(k1v + 2*k2v + 2*k3v + k4v), v0)
            mass = jnp.where(active_now, jnp.maximum(m0 + dt/6*(k1m + 2*k2m + 2*k3m + k4m), 1e-20), m0)
            radius = (mass / (((4. / 3.) * jnp.pi) * config.initial_density)) ** (1. / 3.)
            acc, fg, fd0, fd, fe, fc, fc_pair, fw, fw_pair, fa, air_velocity, fnet = diagnostics
            force_mass, force_radius, force_acceleration = m0, radius_previous, acc

        # Greedy, deterministic hard-contact pairing. Each droplet can merge at
        # most once per timestep; the lower ID survives and carries the union
        # of both lineages. Fixed array shapes keep this operation JIT/GPU safe.
        separation_after = pos[:, None, :] - pos[None, :, :]
        distance2_after = jnp.sum(separation_after**2, axis=2)
        # A tolerance of 0.10 requires 10% interpenetration of the summed
        # radii before permanent coalescence. This avoids treating a grazing
        # numerical contact as an automatic merge.
        contact_distance = (
            (1.0 - config.coalescence_overlap_tolerance)
            * (radius[:, None] + radius[None, :])
        )
        contact2 = contact_distance**2
        overlaps = (
            active_now[:, None]
            & active_now[None, :]
            # Permanent merging is intentionally restricted to two charged
            # droplets. Neutral guards never merge with any droplet.
            & coalescence_eligible[:, None]
            & coalescence_eligible[None, :]
            & jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
            & (distance2_after <= contact2)
        )
        if not config.coalescence_enabled:
            overlaps = jnp.zeros_like(overlaps)

        def choose_partner(i, state):
            partners, used = state
            valid = overlaps[i] & ~used
            has_partner = ~used[i] & jnp.any(valid)
            partner = jnp.argmax(valid).astype(jnp.int32)
            partners = partners.at[i].set(jnp.where(has_partner, partner, -1))
            used = used.at[i].set(used[i] | has_partner)
            used = used.at[partner].set(used[partner] | has_partner)
            return partners, used

        merge_partner, _ = jax.lax.fori_loop(
            0,
            n,
            choose_partner,
            (jnp.full(n, -1, jnp.int32), jnp.zeros(n, bool)),
        )

        def merge_pair(i, state):
            pos_m, vel_m, active_m, mass_m, radius_m, charge_m, acc_m, lineage_m = state
            partner = merge_partner[i]
            merges = partner >= 0
            j = jnp.maximum(partner, 0)
            combined_mass = mass_m[i] + mass_m[j]
            safe_mass = jnp.maximum(combined_mass, 1e-20)
            combined_pos = (mass_m[i] * pos_m[i] + mass_m[j] * pos_m[j]) / safe_mass
            combined_vel = (mass_m[i] * vel_m[i] + mass_m[j] * vel_m[j]) / safe_mass
            combined_acc = (mass_m[i] * acc_m[i] + mass_m[j] * acc_m[j]) / safe_mass
            combined_radius = (safe_mass / (((4.0 / 3.0) * jnp.pi) * config.initial_density)) ** (1.0 / 3.0)
            pos_m = pos_m.at[i].set(jnp.where(merges, combined_pos, pos_m[i]))
            vel_m = vel_m.at[i].set(jnp.where(merges, combined_vel, vel_m[i]))
            acc_m = acc_m.at[i].set(jnp.where(merges, combined_acc, acc_m[i]))
            mass_m = mass_m.at[i].set(jnp.where(merges, combined_mass, mass_m[i]))
            radius_m = radius_m.at[i].set(jnp.where(merges, combined_radius, radius_m[i]))
            charge_m = charge_m.at[i].set(jnp.where(merges, charge_m[i] + charge_m[j], charge_m[i]))
            lineage_m = lineage_m.at[i].set(jnp.where(merges, lineage_m[i] | lineage_m[j], lineage_m[i]))
            pos_m = pos_m.at[j].set(jnp.where(merges, combined_pos, pos_m[j]))
            vel_m = vel_m.at[j].set(jnp.where(merges, jnp.zeros(3), vel_m[j]))
            acc_m = acc_m.at[j].set(jnp.where(merges, jnp.zeros(3), acc_m[j]))
            mass_m = mass_m.at[j].set(jnp.where(merges, 0.0, mass_m[j]))
            radius_m = radius_m.at[j].set(jnp.where(merges, 0.0, radius_m[j]))
            charge_m = charge_m.at[j].set(jnp.where(merges, 0.0, charge_m[j]))
            lineage_m = lineage_m.at[j].set(jnp.where(merges, jnp.zeros(n, bool), lineage_m[j]))
            active_m = active_m.at[j].set(jnp.where(merges, False, active_m[j]))
            return pos_m, vel_m, active_m, mass_m, radius_m, charge_m, acc_m, lineage_m

        pos, vel, active, mass, radius, charge, acc, lineage = jax.lax.fori_loop(
            0,
            n,
            merge_pair,
            (pos, vel, active, mass, radius, charge, acc, lineage),
        )
        active_after_merge = active & released
        charged_after_merge = jnp.abs(charge) > jnp.float32(1e-15)

        coupled_end_z = jnp.where(
            config.terminate_at_field_exit,
            field_upper[2],
            jnp.float32(config.coupled_substrate_z_mm * 1e-3),
        )
        reached_coupled_end = active_after_merge & (pos[:, 2] >= coupled_end_z)
        new_landed = jnp.where(coupled, reached_coupled_end & charged_after_merge, active_after_merge & charged_after_merge & (pos[:, 1] <= config.landing_y_position))
        new_discarded = jnp.where(coupled, reached_coupled_end & ~charged_after_merge, active_after_merge & ~charged_after_merge & (pos[:, 1] <= config.gutter_y_position))
        offset = substrate * (step_index + 1).astype(jnp.float32) * dt
        if coupled:
            # The coupled jet flies along +Z, so the paper plane is XY.
            event = jnp.stack(
                (pos[:, 0] - offset[0], pos[:, 1] - offset[1], jnp.zeros(n, jnp.float32)),
                axis=1,
            )
        else:
            # The legacy jet flies along -Y, so its substrate plane is XZ.
            event = jnp.stack(
                (pos[:, 0] - offset[0], pos[:, 2] - offset[2], jnp.zeros(n, jnp.float32)),
                axis=1,
            )
        landing_pos = jnp.where((new_landed | new_discarded)[:, None], event, landing_pos)
        landed |= new_landed
        discarded |= new_discarded
        active &= ~new_landed & ~new_discarded
        new_carry = (pos, vel, active, landed, discarded, landing_pos, mass, radius, acc, charge, lineage)
        carry = jax.tree.map(lambda new, old: jnp.where(running, new, old), new_carry, old_carry)
        history = {"position_before": old_carry[0], "velocity_before": old_carry[1], "velocity": vel, "force_mass": force_mass, "force_radius": force_radius, "force_charge": old_carry[9], "force_acceleration": force_acceleration, "active_before": active_before, "active_after": active & released, "position": pos, "mass": mass, "radius": radius, "charge": charge, "merge_partner": merge_partner, "acceleration": acc, "Fg": fg, "Fd0": fd0, "Fd": fd, "Fe": fe, "Fc": fc, "Fc_pair": fc_pair, "Fw": fw, "Fw_pair": fw_pair, "Fa": fa, "Uair": air_velocity, "Fnet": fnet}
        return carry, history

    initial = (pos, vel, active, zeros_b, zeros_b, landing_pos, mass0, radius0, acc0, charge, jnp.eye(n, dtype=bool))

    def kernel(charges_coulombs, target_mask):
        state = initial[:9] + (charges_coulombs,) + initial[10:]
        if output_mode == "full":
            final, history = jax.lax.scan(
                lambda carry, step: one_step(carry, step, target_mask),
                state, jnp.arange(steps))
            return final, history
        def endpoint_step(bundle, step):
            carry, landing_time, gutter_x = bundle
            updated, h = one_step(carry, step, target_mask)
            event = (updated[3] & ~carry[3]) | (updated[4] & ~carry[4])
            landing_time = jnp.where(event, (step + 1).astype(jnp.float32) * dt, landing_time)
            before, after = carry[0], updated[0]
            plane = jnp.float32(gutter_z_mm * 1e-3)
            crossing = h["active_before"] & (before[:, 2] < plane) & (after[:, 2] >= plane) & ~jnp.isfinite(gutter_x)
            weight = (plane - before[:, 2]) / jnp.maximum(after[:, 2] - before[:, 2], jnp.float32(1e-20))
            interpolated = before[:, 0] + weight * (after[:, 0] - before[:, 0])
            gutter_x = jnp.where(crossing, interpolated, gutter_x)
            return (updated, landing_time, gutter_x), None
        def condition(bundle):
            step, physical, _, _, _ = bundle
            return (step < steps) & jnp.any(physical[2])
        def advance(bundle):
            step, physical, landing_time, gutter_x, samples = bundle
            old = physical
            (physical, landing_time, gutter_x), _ = endpoint_step((physical, landing_time, gutter_x), step)
            if len(stations):
                times, positions, velocities, masses = samples
                before, after = old[0], physical[0]
                active = old[2] & ((step >= release_steps) if config.sequential_injection else jnp.ones(n, bool))
                # Samples spanning a merge are not interpreted as primary-drop trajectories.
                unmerged = jnp.sum(old[10], axis=1) == jnp.sum(physical[10], axis=1)
                crossing = active[:, None] & unmerged[:, None] & (before[:, 2, None] < station_planes[None, :]) & (after[:, 2, None] >= station_planes[None, :]) & ~jnp.isfinite(times)
                weight = (station_planes[None, :] - before[:, 2, None]) / jnp.maximum(after[:, 2, None] - before[:, 2, None], jnp.float32(1e-20))
                location = before[:, None, :] + weight[:, :, None] * (after-before)[:, None, :]
                velocity = old[1][:, None, :] + weight[:, :, None] * (physical[1]-old[1])[:, None, :]
                mass_sample = old[6][:, None] + weight * (physical[6]-old[6])[:, None]
                times = jnp.where(crossing, (step.astype(jnp.float32) + weight)*dt, times)
                positions = jnp.where(crossing[:, :, None], location, positions)
                velocities = jnp.where(crossing[:, :, None], velocity, velocities)
                masses = jnp.where(crossing, mass_sample, masses)
                samples = times, positions, velocities, masses
            return step + 1, physical, landing_time, gutter_x, samples
        samples = (jnp.full((n,len(stations)),jnp.nan,jnp.float32),jnp.full((n,len(stations),3),jnp.nan,jnp.float32),jnp.full((n,len(stations),3),jnp.nan,jnp.float32),jnp.full((n,len(stations)),jnp.nan,jnp.float32)) if len(stations) else ()
        executed, final, landing_time, gutter_x, samples = jax.lax.while_loop(condition, advance,
            (jnp.int32(0), state, jnp.full(n, jnp.nan, jnp.float32), jnp.full(n, jnp.nan, jnp.float32), samples))
        diagnostics = {"landing_time_s": landing_time, "gutter_x_mm": gutter_x * 1000, "steps_executed": executed}
        if len(stations):
            diagnostics.update(station_time_s=samples[0],station_position_m=samples[1],station_velocity_m_s=samples[2],station_mass_kg=samples[3])
        return final, diagnostics

    return PreparedIntegrator(config, device, output_mode, jax.jit(kernel),
        jax.jit(jax.vmap(kernel, in_axes=(0, 0))), initial_pos, initial[1])


class PreparedIntegrator:
    """Reuse a fixed physical model while varying complete waveform realizations."""
    def __init__(self, config, device, mode, kernel, batch_kernel, position, velocity):
        self.config, self.device, self.output_mode = config, device, mode
        self.kernel, self.batch_kernel = kernel, batch_kernel
        self.initial_position, self.initial_velocity = position, velocity

    def inputs(self, charges_pC=None, target_mask=None):
        cfg = self.config
        q = np.asarray(cfg.droplet_charges_pC if charges_pC is None else charges_pC, dtype=float)
        mask = cfg.intended_print_mask if target_mask is None else target_mask
        mask = np.asarray(np.abs(q) > 1e-12 if mask is None else mask, dtype=bool)
        n = cfg.particles_y_dim
        if q.shape[-1:] != (n,) or q.shape != mask.shape or q.ndim not in (1, 2) or not np.isfinite(q).all():
            raise ValueError("Finite charges and matching target masks must have shape (particles,) or (batch, particles)")
        return (jax.device_put(np.asarray(q * 1e-12, dtype=np.float32), self.device), jax.device_put(mask, self.device))

    def run(self, charges_pC=None, target_mask=None):
        q, mask = self.inputs(charges_pC, target_mask)
        if q.ndim != 1:
            raise ValueError("run() accepts one case; use batch_kernel for batched device output")
        with jax.default_device(self.device):
            final, history = self.kernel(q, mask)
            jax.block_until_ready(final)
        result = {key: np.asarray(jax.device_get(value)) for key, value in history.items()}
        names = ("position", "velocity", "active", "landed", "discarded", "landing_position", "mass", "radius", "acceleration", "charge", "lineage")
        result.update({f"final_{key}": np.asarray(jax.device_get(value)) for key, value in zip(names, final)})
        result["charges"] = np.asarray(jax.device_get(q))
        result["initial_position"] = np.asarray(jax.device_get(self.initial_position))
        result["initial_velocity"] = np.asarray(jax.device_get(self.initial_velocity))
        return result, self.device.platform, str(self.device)


def integrate(config: SimulationConfig, requested_device=None):
    """Compatibility API: complete histories with the original force/event model."""
    return prepare_integrator(config, requested_device, output_mode="full").run()
