"""Public simulation API and host-side result conversion."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SimulationConfig
from .jax_solver import integrate


@dataclass
class Droplet:
    id: int
    mass: float
    radius: float
    charge: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    initial_position: np.ndarray
    landing_position: np.ndarray | None = None
    coalesced_from: list[int] = field(default_factory=list)
    original_jet_index: int = -1

    @property
    def area(self) -> float:
        return np.pi * self.radius**2


@dataclass
class CoalescenceEvent:
    time: float
    survivor_id: int
    absorbed_id: int
    position: np.ndarray
    combined_mass: float
    combined_charge: float
    combined_radius: float
    lineage: list[int]
    survivor_charge_before: float
    absorbed_charge_before: float

    @property
    def collision_type(self) -> str:
        survivor_charged = abs(self.survivor_charge_before) > 1e-15
        absorbed_charged = abs(self.absorbed_charge_before) > 1e-15
        if survivor_charged and absorbed_charged:
            return "charged + charged"
        if survivor_charged or absorbed_charged:
            return "charged + guard"
        return "guard + guard"


@dataclass
class SimulationResult:
    config: SimulationConfig
    droplets: list[Droplet]
    landed_droplets: list[Droplet]
    discarded_droplets: list[Droplet]
    coalescence_events: list[CoalescenceEvent]
    force_history: list[dict]
    pairwise_force_history: list[dict]
    trajectory_data: dict[int, list[dict]]
    xy_snapshot: list[dict]
    xy_snapshot_time: float
    final_time: float
    backend: str
    device: str


def _records(config, data, mask) -> list[Droplet]:
    records = []
    for index in np.where(mask)[0]:
        landing = data["final_landing_position"][index]
        lineage = np.flatnonzero(data["final_lineage"][index]).astype(int).tolist()
        records.append(Droplet(
            id=int(index), mass=float(data["final_mass"][index]),
            radius=float(data["final_radius"][index]),
            charge=float(data["final_charge"][index]),
            position=data["final_position"][index].copy(),
            velocity=data["final_velocity"][index].copy(),
            acceleration=data["final_acceleration"][index].copy(),
            initial_position=data["initial_position"][index].copy(),
            landing_position=landing.copy() if np.any(landing) else None,
            coalesced_from=[source for source in lineage if source != index],
            original_jet_index=config.particles_y_dim - 1 - int(index),
        ))
    return records


def _force_history(config, data, history, final_steps) -> list[dict]:
    rows = []
    stride = max(1, int(round(config.force_capture_interval / config.dt)))
    force_keys = (("Fg_mag_nN", "Fg"), ("Fd0_mag_nN", "Fd0"),
                  ("Fd_mag_nN", "Fd"), ("Fe_mag_nN", "Fe"),
                  ("Fc_mag_nN", "Fc"), ("Fw_mag_nN", "Fw"),
                  ("Fa_mag_nN", "Fa"), ("Fnet_mag_nN", "Fnet"))
    for step in range(0, final_steps, stride):
        for droplet_id in np.where(history["active_before"][step])[0]:
            row = {
                "time": (step + 1) * config.dt, "droplet_id": int(droplet_id),
                "original_jet_index": config.particles_y_dim - 1 - int(droplet_id),
                "position_mm": history["position"][step, droplet_id] * 1000,
                "charge_pC": float(history["charge"][step, droplet_id] * 1e12),
                "mass_pg": float(history["mass"][step, droplet_id] * 1e12),
                "radius_um": float(history["radius"][step, droplet_id] * 1e6),
            }
            row.update({name: float(np.linalg.norm(history[key][step, droplet_id]) * 1e9) for name, key in force_keys})
            row["Uair_mag_mps"] = float(np.linalg.norm(history["Uair"][step, droplet_id]))
            row["Uair_vector_mps"] = history["Uair"][step, droplet_id].copy()
            rows.append(row)
    return rows


def _pairwise_force_history(config, history, final_steps) -> list[dict]:
    rows = []
    stride = max(1, int(round(config.force_capture_interval / config.dt)))
    n = config.particles_y_dim
    for step in range(0, final_steps, stride):
        active = history["active_before"][step]
        for receiver_id in np.where(active)[0]:
            for sender_id in np.where(active)[0]:
                if receiver_id == sender_id:
                    continue
                fc = history["Fc_pair"][step, receiver_id, sender_id]
                fw = history["Fw_pair"][step, receiver_id, sender_id]
                if not np.any(fc) and not np.any(fw):
                    continue
                rows.append({
                    "time": (step + 1) * config.dt,
                    "receiver_id": int(receiver_id),
                    "sender_id": int(sender_id),
                    "receiver_drop": int(receiver_id + 1),
                    "sender_drop": int(sender_id + 1),
                    "receiver_original_jet_index": n - 1 - int(receiver_id),
                    "sender_original_jet_index": n - 1 - int(sender_id),
                    "Fc_pair_x_nN": float(fc[0] * 1e9),
                    "Fc_pair_y_nN": float(fc[1] * 1e9),
                    "Fc_pair_z_nN": float(fc[2] * 1e9),
                    "Fc_pair_mag_nN": float(np.linalg.norm(fc) * 1e9),
                    "Fw_pair_x_nN": float(fw[0] * 1e9),
                    "Fw_pair_y_nN": float(fw[1] * 1e9),
                    "Fw_pair_z_nN": float(fw[2] * 1e9),
                    "Fw_pair_mag_nN": float(np.linalg.norm(fw) * 1e9),
                })
    return rows


def _trajectories(config, history) -> dict[int, list[dict]]:
    result = {index: [] for index in range(config.particles_y_dim)}
    for droplet_id in result:
        for step in np.where(history["active_before"][:, droplet_id])[0]:
            position = history["position"][step, droplet_id]
            result[droplet_id].append({
                "time_ms": (step + 1) * config.dt * 1000,
                "x_mm": float(position[0] * 1000),
                "y_mm": float(position[1] * 1000),
                "z_mm": float(position[2] * 1000),
                "charge_pC": float(history["charge"][step, droplet_id] * 1e12),
                "radius_mm": float(history["radius"][step, droplet_id] * 1000),
            })
    return result


def _snapshot(config, data, history, final_steps):
    target = config.xy_snapshot_target_y_position
    if target is None:
        index = min(max(0, int(np.ceil(config.xy_plot_capture_time / config.dt)) - 1), final_steps - 1)
    else:
        candidates = history["active_after"] & (np.abs(history["charge"]) > 1e-15)
        scores = np.min(np.where(candidates, np.abs(history["position"][:, :, 1] - target), np.inf), axis=1)
        index = int(np.argmin(scores)) if np.any(np.isfinite(scores)) else final_steps - 1
    rows = []
    for droplet_id in np.where(history["active_after"][index])[0]:
        position = history["position"][index, droplet_id]
        rows.append({
            "x": float(position[0] * 1000), "y": float(position[1] * 1000),
            "charge": float(history["charge"][index, droplet_id] * 1e12),
            "radius_mm": float(history["radius"][index, droplet_id] * 1000),
        })
    return rows, index


def _coalescence_events(config, data, history, final_steps) -> list[CoalescenceEvent]:
    events: list[CoalescenceEvent] = []
    lineages = [{index} for index in range(config.particles_y_dim)]
    dynamic_charges = np.asarray(data["charges"], dtype=float).copy()
    for step in range(final_steps):
        for survivor_id in np.flatnonzero(history["merge_partner"][step] >= 0):
            absorbed_id = int(history["merge_partner"][step, survivor_id])
            survivor_charge_before = float(dynamic_charges[survivor_id])
            absorbed_charge_before = float(dynamic_charges[absorbed_id])
            dynamic_charges[survivor_id] += dynamic_charges[absorbed_id]
            dynamic_charges[absorbed_id] = 0.0
            lineages[survivor_id] |= lineages[absorbed_id]
            lineages[absorbed_id] = set()
            lineage = sorted(lineages[survivor_id])
            events.append(
                CoalescenceEvent(
                    time=(step + 1) * config.dt,
                    survivor_id=int(survivor_id),
                    absorbed_id=absorbed_id,
                    position=history["position"][step, survivor_id].copy(),
                    combined_mass=float(history["mass"][step, survivor_id]),
                    combined_charge=float(history["charge"][step, survivor_id]),
                    combined_radius=float(history["radius"][step, survivor_id]),
                    lineage=lineage,
                    survivor_charge_before=survivor_charge_before,
                    absorbed_charge_before=absorbed_charge_before,
                )
            )
    return events


def run_simulation(config: SimulationConfig, device: str | None = None) -> SimulationResult:
    """Run the JIT-compiled CIJ model on JAX's default or requested device.

    ``device`` accepts ``None``/``"auto"``, ``"gpu"``, ``"cpu"``, or ``"tpu"``.
    Device arrays are transferred to NumPy only for the returned reporting API.
    """
    config.validate()
    data, platform, device_name = integrate(config, device)
    finished = np.where(~np.any(data["active_after"], axis=1))[0]
    final_steps = int(finished[0] + 1) if finished.size else len(data["active_after"])
    history = {key: value[:final_steps] for key, value in data.items()
               if not key.startswith("final_") and key not in {"charges", "initial_position"}}
    snapshot, snapshot_index = _snapshot(config, data, history, final_steps)
    survivor_mask = np.any(data["final_lineage"], axis=1)
    return SimulationResult(
        config=config,
        droplets=_records(config, data, survivor_mask),
        landed_droplets=_records(config, data, data["final_landed"]),
        discarded_droplets=_records(config, data, data["final_discarded"]),
        coalescence_events=_coalescence_events(config, data, history, final_steps),
        force_history=_force_history(config, data, history, final_steps),
        pairwise_force_history=_pairwise_force_history(config, history, final_steps),
        trajectory_data=_trajectories(config, history),
        xy_snapshot=snapshot, xy_snapshot_time=(snapshot_index + 1) * config.dt,
        final_time=final_steps * config.dt,
        backend=f"jax-{platform}", device=device_name,
    )
