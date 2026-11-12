"""Shared publication palette and typography; no numerical/model changes."""
import matplotlib as mpl
INK='#203746'
BLUE='#21658A'
TEAL='#238B7E'
ORANGE='#CB7836'
SLATE='#7A8994'
PALE='#EAF1F5'
GRID='#DCE5EB'

def apply():
 mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
  'axes.titlesize':10,'axes.titleweight':'normal','axes.titlepad':12,
  'axes.labelsize':9,'axes.labelpad':7,'axes.edgecolor':INK,'axes.labelcolor':INK,
  'text.color':INK,'xtick.color':INK,'ytick.color':INK,
  'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
  'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.75,
  'axes.prop_cycle':mpl.cycler(color=[BLUE,TEAL,ORANGE,SLATE]),
  'grid.color':GRID,'grid.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,
  'figure.dpi':160,'savefig.pad_inches':.12,'legend.frameon':False})
