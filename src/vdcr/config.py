from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def field_domain_end_z_mm(field_path) -> float:
    """Substrate lies at the last Z coordinate of the imported field domain."""
    with np.load(field_path) as data:
        return float((float(data['origin'][2]) + (data['E'].shape[2]-1)*float(data['spacing'][2]))*1000.)


def build_charge_train() -> np.ndarray:
    return np.array(
        [
            -1.4,
            0.0,
            -1.3,
            0.0,
            -1.2,
            0.0,
            -1.1,
            0.0,
            -1.0,
            0.0,
            -0.9,
            0.0,
            -0.8,
            0.0,
            -0.7,
            0.0,
            -0.6,
            0.0,
            -0.5,
            0.0,
        ],
        dtype=float,
    )


def build_custom_charge_train(
    charged_values_pC,
    guards_per_charged: int | list[int] | np.ndarray = 1,
    trailing_guards: int | None = None,
    leading_guards: int = 0,
) -> np.ndarray:
    """Interleave charges with uniform or independently specified neutral guards.

    A scalar guard count retains the packet convention and is applied after
    every charge. A sequence supplies one count for each gap between charges;
    ``trailing_guards`` then controls guards after the final charge.
    """
    charges = np.asarray(charged_values_pC, dtype=float)
    if charges.ndim != 1 or charges.size == 0:
        raise ValueError("At least one charged-droplet value is required.")
    if not np.all(np.isfinite(charges)) or np.any(np.abs(charges) <= 1e-12):
        raise ValueError("Charged-droplet values must be finite and non-zero.")
    if np.isscalar(guards_per_charged):
        if not isinstance(guards_per_charged, (int, np.integer)) or guards_per_charged < 0:
            raise ValueError("guards_per_charged must be a non-negative integer.")
        gap_guards = np.full(max(0, charges.size - 1), int(guards_per_charged), dtype=int)
        final_guards = int(guards_per_charged) if trailing_guards is None else trailing_guards
    else:
        gap_guards = np.asarray(guards_per_charged)
        if gap_guards.shape != (max(0, charges.size - 1),):
            raise ValueError("Variable guards must provide one count for every gap between charges.")
        final_guards = 0 if trailing_guards is None else trailing_guards
    if not np.issubdtype(gap_guards.dtype, np.integer) or np.any(gap_guards < 0):
        raise ValueError("Every guard count must be a non-negative integer.")
    if not isinstance(final_guards, (int, np.integer)) or final_guards < 0:
        raise ValueError("trailing_guards must be a non-negative integer.")
    if not isinstance(leading_guards, (int, np.integer)) or leading_guards < 0:
        raise ValueError("leading_guards must be a non-negative integer.")
    train: list[float] = [0.0] * int(leading_guards)
    for index, charge in enumerate(charges):
        train.append(float(charge))
        count = int(gap_guards[index]) if index < charges.size - 1 else int(final_guards)
        train.extend([0.0] * count)
    return np.asarray(train, dtype=float)


@dataclass(slots=True)
class SimulationConfig:
    initial_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    nozzle_frequency_hz: float = 76.8e3
    initial_x_velocity: float = 0.0
    initial_y_velocity: float = -24.2
    initial_z_velocity: float = 0.0
    coulomb_constant: float = 8.9875e9
    coulomb_interaction_enabled: bool = True
    coulomb_cutoff_distance_mm: float | None = None
    plate_voltage_diff: float = 8000.0
    pre_deflection_travel_mm: float = 12.0
    primary_deflection_length_mm: float = 18.0
    secondary_deflection_length_mm: float = 13.0
    straight_plate_distance_mm: float = 3.0
    secondary_plate_angle_degrees: float = 12.0
    gutter_travel_mm: float = 10.0
    slot_travel_mm: float = 21.0
    print_height_mm: float = 12.0
    wall_velocity_x: float = 0.0
    wall_velocity_y: float = 0.0
    wall_velocity_z: float = 0.0
    particle_diameter_mm: float = 0.13
    droplet_density_kg_per_m3: float = 870.0
    air_density: float = 1.225
    air_dynamic_viscosity_pa_s: float = 1.81e-5
    # Reduced-order positive-air model. Volumetric port flow is converted to
    # an entrained axial co-flow through Q/A and scaled by a coupling factor.
    positive_air_flow_lpm: float = 0.0
    positive_air_effective_area_mm2: float = 10.0
    positive_air_coupling_efficiency: float = 0.50
    positive_air_unsteady_enabled: bool = True
    positive_air_reference_flow_lpm: float = 3.0
    positive_air_crossflow_rms_mps: float = 1.0
    positive_air_axial_rms_mps: float = 0.40
    positive_air_vortex_strength_mps: float = 1.5
    positive_air_vortex_frequency_hz: float = 450.0
    positive_air_correlation_time_s: float = 0.50e-3
    positive_air_correlation_length_m: float = 3.0e-3
    positive_air_random_modes: int = 4
    positive_air_seed: int = 1729
    gravity_acceleration: float = 9.81
    wake_max_axial_length_factor: float = 100.0
    wake_min_axial_length_factor: float = 2.0
    wake_spread_angle_degrees: float = 10.0
    wake_initial_velocity_deficit_factor: float = 0.25
    wake_radial_gaussian_coefficient: float = 4.605
    wake_max_velocity_deficit_fraction: float = 0.12
    body_force_x: float = 0.0
    body_force_z: float = 0.0
    electric_field_z_ratio: float = 0.0
    openfoam_field_path: str | None = None
    sequential_injection: bool = False
    terminate_at_field_exit: bool = False
    coupled_substrate_z_mm: float = 64.23000208418071
    coalescence_enabled: bool = True
    coalescence_overlap_tolerance: float = 0.10
    particles_x_dim: int = 1
    particles_y_dim: int = 20
    droplet_charges_pC: np.ndarray = field(
        default_factory=build_charge_train
    )
    # Optional controller-level distinction between intended printed pulses
    # and nonzero compensation pulses in imported voltage waveforms.
    intended_print_mask: np.ndarray | None = None
    dt: float = 1e-6
    max_simulation_time: float = 5e-3
    xy_plot_capture_time: float = 1.6e-3
    xy_snapshot_target_y_mm: float | None = -60.0
    force_capture_interval: float = 0.1e-3
    raster_repeat_count: int = 1
    raster_droplets_per_repeat: int = 0

    @property
    def y_spacing(self) -> float:
        return abs(self.initial_y_velocity) / self.nozzle_frequency_hz

    @property
    def wake_spread_angle_rad(self) -> float:
        return np.deg2rad(self.wake_spread_angle_degrees)

    @property
    def initial_droplet_radius(self) -> float:
        return self.particle_diameter_mm / 2 / 1000

    @property
    def initial_particle_volume(self) -> float:
        return (4 / 3) * np.pi * self.initial_droplet_radius**3

    @property
    def initial_particle_mass(self) -> float:
        return self.droplet_density_kg_per_m3 * self.initial_particle_volume

    @property
    def initial_density(self) -> float:
        return self.droplet_density_kg_per_m3

    @property
    def mass_loss_rate_per_meter(self) -> float:
        return (0.01 * self.initial_particle_mass) / 0.010

    @property
    def total_particles(self) -> int:
        return self.particles_x_dim * self.particles_y_dim

    @property
    def droplet_charges_coulombs(self) -> np.ndarray:
        return self.droplet_charges_pC * 1e-12

    @property
    def wake_max_axial_length(self) -> float:
        return self.wake_max_axial_length_factor * self.initial_droplet_radius

    @property
    def deflection_plate_start_y(self) -> float:
        return -self.pre_deflection_travel_mm * 1e-3

    @property
    def primary_deflection_end_y(self) -> float:
        return -(
            self.pre_deflection_travel_mm + self.primary_deflection_length_mm
        ) * 1e-3

    @property
    def secondary_deflection_end_y(self) -> float:
        return -(
            self.pre_deflection_travel_mm
            + self.primary_deflection_length_mm
            + self.secondary_deflection_length_mm
        ) * 1e-3

    @property
    def landing_y_position(self) -> float:
        return self.secondary_deflection_end_y - (
            self.slot_travel_mm + self.print_height_mm
        ) * 1e-3

    @property
    def gutter_y_position(self) -> float:
        return self.secondary_deflection_end_y - self.gutter_travel_mm * 1e-3

    @property
    def slot_y_position(self) -> float:
        return self.secondary_deflection_end_y - self.slot_travel_mm * 1e-3

    @property
    def xy_snapshot_target_y_position(self) -> float | None:
        if self.xy_snapshot_target_y_mm is None:
            return None
        return self.xy_snapshot_target_y_mm * 1e-3

    @property
    def substrate_velocity_vector(self) -> np.ndarray:
        return np.array(
            [self.wall_velocity_x, self.wall_velocity_y, self.wall_velocity_z],
            dtype=float,
        )

    @property
    def body_force_vector(self) -> np.ndarray:
        return np.array(
            [self.body_force_x, -self.gravity_acceleration, self.body_force_z],
            dtype=float,
        )

    @property
    def positive_air_speed_m_per_s(self) -> float:
        """Effective entrained co-flow speed from the selected port flow."""
        flow_m3_per_s = self.positive_air_flow_lpm * 1e-3 / 60.0
        area_m2 = self.positive_air_effective_area_mm2 * 1e-6
        return self.positive_air_coupling_efficiency * flow_m3_per_s / area_m2

    @property
    def positive_air_velocity_vector(self) -> np.ndarray:
        """Mean axial carrier-air velocity; +Z coupled or -Y legacy flight."""
        speed = self.positive_air_speed_m_per_s
        if self.openfoam_field_path is not None:
            return np.array([0.0, 0.0, speed], dtype=float)
        return np.array([0.0, -speed, 0.0], dtype=float)

    @property
    def positive_air_unsteady_scale(self) -> float:
        """Scale reference unsteady amplitudes with the selected flow rate."""
        if not self.positive_air_unsteady_enabled:
            return 0.0
        return self.positive_air_flow_lpm / self.positive_air_reference_flow_lpm

    @property
    def straight_plate_distance(self) -> float:
        return self.straight_plate_distance_mm * 1e-3

    @property
    def secondary_plate_angle_rad(self) -> float:
        return np.deg2rad(self.secondary_plate_angle_degrees)

    def plate_gap_at_y(self, y_position: float) -> float | None:
        if y_position > self.deflection_plate_start_y:
            return None
        if y_position >= self.primary_deflection_end_y:
            return self.straight_plate_distance
        if y_position >= self.secondary_deflection_end_y:
            axial_distance = abs(y_position - self.primary_deflection_end_y)
            return self.straight_plate_distance + axial_distance * np.tan(
                self.secondary_plate_angle_rad
            )
        return None

    def electric_field_vector_at_y(self, y_position: float) -> np.ndarray:
        gap = self.plate_gap_at_y(y_position)
        if gap is None:
            return np.zeros(3, dtype=float)
        field_x = self.plate_voltage_diff / gap
        return np.array(
            [field_x, 0.0, field_x * self.electric_field_z_ratio],
            dtype=float,
        )

    def validate(self) -> None:
        if len(self.droplet_charges_pC) != self.particles_y_dim:
            raise ValueError(
                "Number of assigned charges must match particles_y_dim "
                f"({len(self.droplet_charges_pC)} != {self.particles_y_dim})."
            )
        if self.intended_print_mask is not None:
            intended = np.asarray(self.intended_print_mask, dtype=bool)
            if intended.shape != (self.particles_y_dim,):
                raise ValueError("intended_print_mask must contain one flag per particle.")
        if self.straight_plate_distance_mm <= 0:
            raise ValueError("straight_plate_distance_mm must be positive.")
        if self.coulomb_cutoff_distance_mm is not None:
            if not np.isfinite(self.coulomb_cutoff_distance_mm) or self.coulomb_cutoff_distance_mm <= 0:
                raise ValueError("coulomb_cutoff_distance_mm must be positive or None.")
        if self.air_dynamic_viscosity_pa_s <= 0:
            raise ValueError("air_dynamic_viscosity_pa_s must be positive.")
        if not np.isfinite(self.positive_air_flow_lpm) or self.positive_air_flow_lpm < 0:
            raise ValueError("positive_air_flow_lpm must be finite and non-negative.")
        if not np.isfinite(self.positive_air_effective_area_mm2) or self.positive_air_effective_area_mm2 <= 0:
            raise ValueError("positive_air_effective_area_mm2 must be finite and positive.")
        if not 0.0 <= self.positive_air_coupling_efficiency <= 1.0:
            raise ValueError("positive_air_coupling_efficiency must be in [0, 1].")
        if not np.isfinite(self.positive_air_reference_flow_lpm) or self.positive_air_reference_flow_lpm <= 0:
            raise ValueError("positive_air_reference_flow_lpm must be finite and positive.")
        if not np.isfinite(self.positive_air_crossflow_rms_mps) or self.positive_air_crossflow_rms_mps < 0:
            raise ValueError("positive_air_crossflow_rms_mps must be finite and non-negative.")
        if not np.isfinite(self.positive_air_axial_rms_mps) or self.positive_air_axial_rms_mps < 0:
            raise ValueError("positive_air_axial_rms_mps must be finite and non-negative.")
        if not np.isfinite(self.positive_air_vortex_strength_mps) or self.positive_air_vortex_strength_mps < 0:
            raise ValueError("positive_air_vortex_strength_mps must be finite and non-negative.")
        if not np.isfinite(self.positive_air_vortex_frequency_hz) or self.positive_air_vortex_frequency_hz < 0:
            raise ValueError("positive_air_vortex_frequency_hz must be finite and non-negative.")
        if not np.isfinite(self.positive_air_correlation_time_s) or self.positive_air_correlation_time_s <= 0:
            raise ValueError("positive_air_correlation_time_s must be finite and positive.")
        if not np.isfinite(self.positive_air_correlation_length_m) or self.positive_air_correlation_length_m <= 0:
            raise ValueError("positive_air_correlation_length_m must be finite and positive.")
        if not isinstance(self.positive_air_random_modes, (int, np.integer)) or not 1 <= self.positive_air_random_modes <= 16:
            raise ValueError("positive_air_random_modes must be an integer from 1 to 16.")
        if not isinstance(self.positive_air_seed, (int, np.integer)) or self.positive_air_seed < 0:
            raise ValueError("positive_air_seed must be a non-negative integer.")
        if not np.all(np.isfinite(self.substrate_velocity_vector)):
            raise ValueError("Substrate velocity components must be finite.")
        if not isinstance(self.raster_repeat_count, (int, np.integer)) or self.raster_repeat_count < 1:
            raise ValueError("raster_repeat_count must be a positive integer.")
        if not isinstance(self.raster_droplets_per_repeat, (int, np.integer)) or self.raster_droplets_per_repeat < 0:
            raise ValueError("raster_droplets_per_repeat must be a non-negative integer.")
        if self.raster_droplets_per_repeat and self.raster_droplets_per_repeat * self.raster_repeat_count != self.particles_y_dim:
            raise ValueError("Raster repeat metadata must match the total droplet count.")
        if self.wake_radial_gaussian_coefficient <= 0:
            raise ValueError("wake_radial_gaussian_coefficient must be positive.")
        if self.wake_min_axial_length_factor < 0:
            raise ValueError("wake_min_axial_length_factor must be non-negative.")
        if self.wake_max_axial_length_factor <= self.wake_min_axial_length_factor:
            raise ValueError("wake_max_axial_length_factor must exceed the minimum.")
        if not 0.0 <= self.wake_initial_velocity_deficit_factor <= 1.0:
            raise ValueError("wake_initial_velocity_deficit_factor must be in [0, 1].")
        if not 0.0 <= self.wake_max_velocity_deficit_fraction < 1.0:
            raise ValueError("wake_max_velocity_deficit_fraction must be in [0, 1).")
        if not 0.0 <= self.coalescence_overlap_tolerance < 1.0:
            raise ValueError("coalescence_overlap_tolerance must be in [0, 1).")
        if self.openfoam_field_path is not None and self.initial_z_velocity <= 0:
            raise ValueError("OpenFOAM-coupled flight requires positive initial_z_velocity.")
        if self.openfoam_field_path is not None and self.coupled_substrate_z_mm <= self.initial_position_mm[2]:
            raise ValueError("Coupled substrate Z must be downstream of the injection point.")


def default_config() -> SimulationConfig:
    config = SimulationConfig()
    config.validate()
    return config


def openfoam_coupled_config(field_path: str) -> SimulationConfig:
    """Configure sequential +Z injection through an OpenFOAM field."""
    # Emit 34 charged droplets, each followed by one neutral guard. Charge
    # magnitude increases with release order from 0.60 pC to 1.92 pC.
    charged_pC = -(0.60 + 0.04 * np.arange(34, dtype=float))
    coupled_charge_train = build_custom_charge_train(charged_pC, guards_per_charged=1)
    config = SimulationConfig(
        initial_position_mm=(0.0, 0.0, 0.130),
        initial_y_velocity=0.0,
        initial_z_velocity=24.2,
        openfoam_field_path=field_path,
        sequential_injection=True,
        terminate_at_field_exit=False,
        coupled_substrate_z_mm=field_domain_end_z_mm(field_path),
        particles_y_dim=68,
        droplet_charges_pC=coupled_charge_train,
        max_simulation_time=6e-3,
        xy_snapshot_target_y_mm=None,
    )
    config.validate()
    return config
