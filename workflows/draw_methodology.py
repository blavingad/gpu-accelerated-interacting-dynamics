"""Render the reusable-field simulation and inverse-design schematic."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from vdcr.paper_style import apply,INK,BLUE,TEAL,PALE
OUT=Path(__file__).resolve().parents[1]/'paper/figures'
apply()
fig,ax=plt.subplots(figsize=(11,4.6));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,4.6))
def node(x,y,w,h,head,body,color=PALE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor=color,edgecolor=TEAL,lw=1));ax.text(x+w/2,y+h-.28,head,ha='center',va='center',weight='bold',fontsize=10,color=INK);ax.text(x+w/2,y+.46,body,ha='center',va='center',fontsize=9,color=INK)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':INK,'lw':1.3})
node(.15,2.8,2.9,1.3,'Model inputs','Device, ink and electronics\noperating envelope')
node(4.1,2.8,2.9,1.3,'Reusable external field','Solve once per configuration\nspatial field representation')
node(8,2.8,2.8,1.3,'Design requirements','Targets, speed and resolution\nactuation and clearance limits')
node(.15,.65,2.9,1.3,'Candidate structure','Order + guards + commands\ncontinuous actuator history')
node(4.1,.65,2.9,1.3,'Interacting particle dynamics','Pair forces + directed wakes\ncarrier air + masked events')
node(8,.65,2.8,1.3,'Constrained inverse design','Joint charge + dependency fitting\ninteger-command replay')
arrow((3.12,3.45),(4.02,3.45));arrow((5.55,2.75),(5.55,2.03));arrow((9.4,2.75),(9.4,2.03));arrow((3.12,1.3),(4.02,1.3));arrow((7.08,1.3),(7.92,1.3));ax.add_patch(FancyArrowPatch(path=MplPath([(9.4,.56),(9.4,.30),(1.6,.30),(1.6,.56)]),arrowstyle='->',mutation_scale=10,lw=1.1,color=TEAL));ax.text(5.5,.08,'Digital design loop: structure, charges and dependencies; device field reused across candidates',ha='center',fontsize=9,color=INK);fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(OUT/f'methodology_overview.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
