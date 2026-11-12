from pathlib import Path
import unittest,json,hashlib,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'workflows')]
from vdcr.framework_workload import Workload
from vdcr.charging_sequence import run_sequence
from fit_shared_charges import realized_charges

class ReproductionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.models={case:Workload(case) for case in ['low','high']}
 def test_runtime_is_local(self):
  import vdcr,vdcr.jax_solver
  for module in [vdcr,vdcr.jax_solver]:self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT))
 def test_declared_context_domain(self):
  for w in self.models.values():
   self.assertEqual(len(w.pairs),644);self.assertEqual(sum(p==0 for p,c in w.pairs),235)
   self.assertEqual(len(w.selected_pairs(''.join(chr(i) for i in range(32,127)))),644)
   self.assertTrue(Path(w.config.openfoam_field_path).is_relative_to(ROOT))
 def test_frozen_paper_euler_snapshot(self):
  digest=hashlib.sha256((ROOT/'tests/fixtures/paper_euler_solver.py').read_bytes()).hexdigest()
  self.assertEqual(digest,json.loads((ROOT/'results/reference/low_complete_pool/completed.json').read_text())['model_sha256'])
 def test_waveform_and_inverse(self):
  for w in self.models.values():
   ids=[w.pair_index[(0,1)],w.pair_index[(65535,65535)]]
   for index in ids:
    u=w.events(w.full_commands,[index])[0]
    seq=run_sequence(u,u!=0,w.params,compensation_fractions=(0.,0.),induction_coefficients=(.1,.02),breakup_phase=.5)
    self.assertLess(np.max(abs(w.charges(w.full_commands,[index])[0]-np.array([r['assigned_charge_pC'] for r in seq['trace']]))),1e-11)
   table=np.array([w.lookup['entries'][str(d)][a] for d,a in w.keys]);actual=realized_charges(w,table,ids)
   for j,index in enumerate(ids):
    keys=w.indices[index];on=keys>=0;self.assertTrue(np.allclose(actual[j,on],-table[keys[on]],atol=2e-12,rtol=0))
 def test_measured_coordinate_inputs(self):
  for case in self.models:
   measured=pd.read_csv(ROOT/f'inputs/cases/{case}/measured_coordinates.csv')
   self.assertEqual(len(measured),1725)
   columns=['landing_mm','substrate_horizontal_mm']
   self.assertTrue(np.isfinite(measured[columns].to_numpy()).all())
 def test_current_experimental_assessment(self):
  from vdcr.experimental_data import assessment_directory
  folder=assessment_directory()
  paired=pd.read_csv(folder/'experimental_paired_dots_by_glyph.csv')
  keys=['line','column_index','drop']
  for case,label in [('low','A'),('high','B')]:
   current=pd.read_csv(ROOT/f'inputs/cases/{case}/measured_coordinates.csv')
   subset=paired[paired.configuration==label]
   joined=current.merge(subset,on=keys,validate='one_to_one')
   self.assertEqual(len(joined),1725)
   np.testing.assert_allclose(joined.landing_mm,joined.measured_deflection_mm,atol=1e-12,rtol=0)
   np.testing.assert_allclose(joined.substrate_horizontal_mm,joined.measured_horizontal_mm,atol=1e-12,rtol=0)
  if (folder/'frozen.json').exists():
   for path,digest in json.loads((folder/'frozen.json').read_text())['files'].items():
    self.assertEqual(hashlib.sha256((folder/path).read_bytes()).hexdigest(),digest)
 def test_native_smokes_match_reference_when_available(self):
  for case in self.models:
   path=ROOT/f'results/generated/{case}_smoke/positions.csv'
   if not path.exists():continue
   new=pd.read_csv(path);ref=pd.read_csv(ROOT/f'results/reference/{case}_complete_pool/positions.csv');key=['previous_mask','current_mask','drop']
   j=new.merge(ref,on=key,suffixes=('_new','_ref'),validate='one_to_one');self.assertEqual(len(j),len(new))
   self.assertLess(np.max(abs(j.landing_mm_new-j.landing_mm_ref))*1000,.05)
if __name__=='__main__':unittest.main()
