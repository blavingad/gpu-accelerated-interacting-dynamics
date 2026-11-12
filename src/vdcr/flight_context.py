import numpy as np
def landing_by_current_drop(result, event_print_drop: np.ndarray, stroke_index: np.ndarray, intended: np.ndarray) -> dict[int, dict]:
    current_event_ids = set(np.flatnonzero((stroke_index == 1) & intended).tolist())
    printed_drops = sorted(int(event_print_drop[event_id]) for event_id in current_event_ids)
    rows = {
        drop: {"landed": False, "x_mm": np.nan, "y_mm": np.nan, "z_mm": np.nan, "lineage_size": 0}
        for drop in printed_drops
    }
    for droplet in result.landed_droplets:
        if droplet.landing_position is None:
            continue
        lineage = {droplet.id, *droplet.coalesced_from}
        for event_id in sorted(lineage.intersection(current_event_ids)):
            drop = int(event_print_drop[event_id])
            rows[drop] = {
                "landed": True,
                "x_mm": float(droplet.landing_position[0] * 1000.0),
                "y_mm": float(droplet.landing_position[1] * 1000.0),
                "z_mm": float(droplet.landing_position[2] * 1000.0),
                "lineage_size": int(len(lineage)),
            }
    return rows

def current_coalesced_print_drops(result, event_print_drop: np.ndarray, stroke_index: np.ndarray) -> set[int]:
    drops: set[int] = set()
    for event in result.coalescence_events:
        for event_id in event.lineage:
            if int(stroke_index[event_id]) != 1:
                continue
            drop = int(event_print_drop[event_id])
            if drop > 0:
                drops.add(drop)
    return drops
