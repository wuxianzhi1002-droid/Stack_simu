"""Create compact, hash-audited remote input bundles without altering full local data."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha256(p):
 d=hashlib.sha256()
 with p.open('rb') as h:
  for b in iter(lambda:h.read(1048576),b''):d.update(b)
 return d.hexdigest()
def copy_fields(src,dst,fields,extra):
 with np.load(src,allow_pickle=False) as d:
  missing=[k for k in fields if k not in d]
  if missing:raise KeyError(f'{src.name}: {missing}')
  payload={k:np.asarray(d[k]) for k in fields};payload.update(extra(d))
 np.savez_compressed(dst,**payload)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage1',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();root=a.output_root.resolve();root.mkdir(parents=True,exist_ok=False);sout=root/'stage1';cout=root/'calibration';sout.mkdir();cout.mkdir()
 stage_fields=['wavelengths','reported_wavelengths_nm','spectrum_measured','config_json','reported_axis_is_fixed','internal_wavelength_margin_nm','measured_reflector_angle_deg','angle_measurement_sigma_deg','angle_measurement_mode','noise_case','noise_factor','noise_level','realization_index','random_seed','generator_version','optical_backend','layer_names','layer_thickness_um','true_reflector_angle_deg','noise_realization_json']
 srows=[]
 for src in sorted(a.stage1.glob('static_spectrum_*_typical_*.npz')):
  dst=sout/src.name;copy_fields(src,dst,stage_fields,lambda d:{'full_local_source_filename':np.asarray(src.name),'full_local_source_sha256':np.asarray(sha256(src))});srows.append({'filename':dst.name,'sha256':sha256(dst),'bytes':dst.stat().st_size,'full_source_sha256':sha256(src)})
 cal_fields=['reported_wavelengths_nm','spectra_measured','multiframe_provenance','multiframe_source_filename','multiframe_source_sha256','generator_version','angle_measurement_sigma_deg']
 crows=[]
 for src in sorted(a.calibration.glob('multiframe_g*.npz')):
  dst=cout/src.name;copy_fields(src,dst,cal_fields,lambda d:{'full_local_source_filename':np.asarray(src.name),'full_local_source_sha256':np.asarray(sha256(src))});crows.append({'filename':dst.name,'sha256':sha256(dst),'bytes':dst.stat().st_size,'full_source_sha256':sha256(src)})
 manifest={'version':'v15_compact_remote_inputs','stage1_count':len(srows),'calibration_count':len(crows),'stage1':srows,'calibration':crows,'scope':'Lossless copies of every field consumed by V15 remote runners; complete StackRT audit arrays remain in full local datasets.'}
 (cout/'multiframe_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');(root/'compact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 if len(srows)!=100 or len(crows)!=10:raise RuntimeError((len(srows),len(crows)))
 print(json.dumps({'overall':'PASS','root':str(root),'stage1_bytes':sum(x['bytes'] for x in srows),'calibration_bytes':sum(x['bytes'] for x in crows)},indent=2))
if __name__=='__main__':main()
