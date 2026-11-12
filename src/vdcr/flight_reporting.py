from pathlib import Path
from types import SimpleNamespace
import numpy as np
GUTTER_Z_MM=43.770
MINIMUM_DEFLECTION_MM=.5
def crossings(data, event_ids, *, z_mm=GUTTER_Z_MM):
    """Interpolate the first downstream crossing; missing crossings stay NaN."""
    result=[]
    for event in event_ids:
        pos=np.asarray(data['position'][:,event])*1000
        hits=np.flatnonzero(np.asarray(data['active_before'][:,event]) & (pos[:,2]>=z_mm))
        x=np.nan
        if len(hits):
            k=int(hits[0]);a=pos[k-1] if k else np.asarray(data['initial_position'][event])*1000;b=pos[k]
            if b[2]>a[2]:x=float(a[0]+(z_mm-a[2])/(b[2]-a[2])*(b[0]-a[0]))
        result.append(dict(event_id=int(event),x_gutter_mm=x,passes=bool(np.isfinite(x) and x>=MINIMUM_DEFLECTION_MM)))
    return result

def run_print_simulation(config, device=None, *, proximity_diagnostics=False, trajectory_path=None):
    """Integrate a context and report deposited positions and gutter crossings.

    Retains native step histories while omitting per-step Python record conversion.
    """
    from .jax_solver import integrate
    from .model import _records, _coalescence_events
    config.validate()
    data,platform,device_name=integrate(config,device)
    finished=np.flatnonzero(~np.any(data['active_after'],axis=1))
    end=int(finished[0]+1) if len(finished) else len(data['active_after'])
    intended=np.asarray(config.intended_print_mask if config.intended_print_mask is not None else np.abs(config.droplet_charges_pC)>0)
    min_gap=np.nan
    if proximity_diagnostics:
        printed=np.flatnonzero(intended)
        minimum=np.inf
        for start in range(0,end,128):
            pos=data['position'][start:start+128];rad=data['radius'][start:start+128];active=data['active_before'][start:start+128]
            delta=pos[:,:,None,:]-pos[:,None,:,:]
            gap=np.sqrt(np.sum(delta*delta,axis=-1))-rad[:,:,None]-rad[:,None,:]
            valid=active[:,:,None]&active[:,None,:]&~np.eye(len(intended),dtype=bool)[None,:,:]
            if len(printed):minimum=min(minimum,float(np.min(np.where(valid[:,printed,:],gap[:,printed,:],np.inf))))
        min_gap=minimum*1000 if np.isfinite(minimum) else np.nan
    if trajectory_path is not None:
        path=Path(trajectory_path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,position_m=data['position'][:end],active=data['active_before'][:end],radius_m=data['radius'][:end],charge_C=data['charge'][:end],merge_partner=data['merge_partner'][:end],dt_s=config.dt,printed_event_mask=intended)
    result=SimpleNamespace(config=config,landed_droplets=_records(config,data,data['final_landed']),
        coalescence_events=_coalescence_events(config,data,data,end),device=device_name,backend='jax-'+platform,
        gutter_crossings=crossings(data,np.flatnonzero(intended)),unprinted_gutter_crossings=crossings(data,np.flatnonzero(~intended)),minimum_printed_surface_gap_mm=min_gap)
    return result
