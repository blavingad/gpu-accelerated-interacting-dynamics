"""Measured/simulated character reconstruction from frozen paired coordinates."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection,LineCollection
from matplotlib.patches import Circle
from .paper_style import apply,BLUE,ORANGE,INK,SLATE
from .experimental_data import assessment_file
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/generated'
SOURCE=assessment_file('experimental_paired_dots_by_glyph.csv')
CASES={'A':('Low-speed raster',50),'B':('High-speed raster',100)}

def load():
 f=pd.read_csv(SOURCE)
 assert len(f)==3450 and not f.duplicated(['configuration','line','column_index','drop']).any()
 assert set(f.glyph)==set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
 assert np.isfinite(f[['measured_horizontal_mm','measured_deflection_mm','simulated_horizontal_mm','simulated_deflection_mm']]).all().all()
 return f

def footprint(ax,f,prefix,diameter):
 ax.add_collection(PatchCollection([Circle((x,y),diameter/2000) for x,y in zip(f[prefix+'_horizontal_mm'],f[prefix+'_deflection_mm'])],facecolor=INK,edgecolor='none'))

def overlay(ax,f,legend=False):
 sim=f[['simulated_horizontal_mm','simulated_deflection_mm']].to_numpy();meas=f[['measured_horizontal_mm','measured_deflection_mm']].to_numpy()
 ax.add_collection(LineCollection(np.stack([sim,meas],axis=1),colors=SLATE,linewidths=.5,zorder=1))
 ax.scatter(*sim.T,s=10,c=BLUE,label='Simulated centers',zorder=2)
 ax.scatter(*meas.T,s=15,marker='x',linewidths=.65,c=ORANGE,label='Experiment · ten-print mean',zorder=3)
 if legend:ax.legend(loc='upper left',bbox_to_anchor=(0,1.35),ncol=2,frameon=False,fontsize=8)

def overview(f,case='A',diameter=528,section='A–Z'):
 apply();name,speed=CASES[case];f=f[f.configuration.eq(case)]
 groups=['ABCDEFGHIJKLM','NOPQRSTUVWXYZ'] if section=='A–Z' else ['ABCDEFGHIJKLM'] if section=='A–M' else ['NOPQRSTUVWXYZ']
 fig,axes=plt.subplots(3*len(groups),1,figsize=(9,4.2*len(groups)),squeeze=False)
 for k,letters in enumerate(groups):
  g=f[f.glyph.isin(list(letters))];x=g[['simulated_horizontal_mm','measured_horizontal_mm']].to_numpy();y=g[['simulated_deflection_mm','measured_deflection_mm']].to_numpy()
  pad=diameter/2000+.15
  for j,(label,prefix) in enumerate([('Simulation','simulated'),('Experiment · ten-print mean','measured'),('Overlay of matched centers','overlay')]):
   ax=axes[3*k+j,0]
   if prefix=='overlay':overlay(ax,g)
   else:footprint(ax,g,prefix,diameter)
   ax.set(xlim=(x.max()+pad,x.min()-pad),ylim=(y.min()-pad,y.max()+pad));ax.set_aspect('equal',adjustable='box');ax.axis('off')
   ax.set_title(f'{letters[0]}–{letters[-1]} · {label}',fontsize=10,pad=5)
 handles,labels=axes[2,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.915),ncol=2,frameon=False,fontsize=8)
 fig.suptitle(f'{name} · {speed} m/min · 75 kHz · 60 DPI\nCoordinate reconstructions · equal millimeter axes',fontsize=12,y=.99)
 fig.text(.5,.012,f'Nominal footprint diameter {diameter:g} µm; measured dot sizes were not supplied. Overlay uses centers.\nNo fitted translation, rotation, scaling, shear, or per-character registration.',ha='center',fontsize=8)
 fig.subplots_adjust(top=.85,bottom=.10,hspace=.95)
 return fig

def closeups(f,letters='AMSZ'):
 apply();fig,axes=plt.subplots(2,len(letters),figsize=(9,7),squeeze=False)
 for i,case in enumerate(CASES):
  for j,ch in enumerate(letters):
   g=f[f.configuration.eq(case)&f.glyph.eq(ch)].copy()
   # One identical plotting-origin subtraction for both coordinate sets, never a registration fit.
   origin=float(g.simulated_horizontal_mm.min())
   for prefix in ['simulated','measured']:g[prefix+'_horizontal_mm']-=origin
   ax=axes[i,j];overlay(ax,g)
   x=g[['simulated_horizontal_mm','measured_horizontal_mm']].to_numpy();y=g[['simulated_deflection_mm','measured_deflection_mm']].to_numpy()
   ax.set(xlim=(x.max()+.2,x.min()-.2),ylim=(y.min()-.2,y.max()+.2));ax.set_aspect('equal',adjustable='box');ax.set_title(f'{CASES[case][0]} · {ch}',fontsize=10)
   if i==1:ax.set_xlabel('Substrate direction [mm]',fontsize=8)
   if j==0:ax.set_ylabel('Deflection [mm]',fontsize=8)
   ax.tick_params(labelsize=7);ax.spines[['top','right']].set_visible(False)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
 fig.suptitle('Matched-coordinate overlays · displacement shown at 1×',fontsize=13)
 fig.text(.5,.015,'A, M, S and Z show different stroke shapes; all A–Z dots are included in the overview and metrics.\nOne common plotting origin per glyph is subtracted from both sets; their relative positions are unchanged.',ha='center',fontsize=8)
 fig.subplots_adjust(left=.075,right=.99,top=.84,bottom=.15,hspace=.48,wspace=.42);return fig

def build():
 f=load();dest=OUT/'figures';dest.mkdir(parents=True,exist_ok=True);paper=ROOT/'paper/figures'
 for case,label in [('A','low'),('B','high')]:
  fig=overview(f,case)
  for path in [dest/f'coordinate_print_{label}.png',paper/f'coordinate_print_{label}.pdf']:fig.savefig(path,bbox_inches='tight',dpi=220)
  plt.close(fig)
 fig=closeups(f)
 for path in [dest/'coordinate_print_closeups.png',paper/'coordinate_print_closeups.pdf']:fig.savefig(path,bbox_inches='tight',dpi=220)
 plt.close(fig)
 (OUT/'coordinate_visualization_manifest.json').write_text(json.dumps(dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),dots=3450,dots_per_case=1725,glyphs_per_case=26,registration='none',footprint='nominal 528 micrometers; not measured diameters',overlay='matched centers and unit-scale connecting residuals',raw_coordinates_unchanged=True),indent=2)+'\n')
 return f

