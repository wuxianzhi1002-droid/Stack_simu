"""Aggregate V15 EQ-99X Stage 1 and Stage 3 paired comparisons."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def load(path,version):
 r=json.loads(path.read_text(encoding='utf-8'))
 if r.get('overall')!='PASS' or r.get('version')!=version:raise ValueError(f'invalid report {path}')
 return r
def stats(x):
 a=np.asarray(x,dtype=float);return {'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(a.max())}
def write_csv(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--stage1-reports',type=Path,nargs=5,required=True);p.add_argument('--stage3-reports',type=Path,nargs=2,required=True);p.add_argument('--output-dir',type=Path,required=True);return p.parse_args()
def main():
 a=parse_args();s1=[load(p,'v15_stage1a_wideband_information') for p in a.stage1_reports];s3=[load(p,'v15_stage3_multiangle_joint_gpu') for p in a.stage3_reports];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 by_band1={r['configuration']['band_label']:r for r in s1};by_band3={r['configuration']['band']:r for r in s3}
 if set(by_band1)!={'220-580','200-600','200-650','200-700','200-800'} or set(by_band3)!={'220-580','200-800'}:raise ValueError('missing bands')
 if by_band3['220-580']['source_dataset_bundle_sha256']!=by_band3['200-800']['source_dataset_bundle_sha256']:raise ValueError('Stage 3 source bundle differs')
 rows=[]
 for band,r in by_band1.items():
  agg=r['aggregate'];rows.append({'stage':'stage1_single','band':band,'mode':'fixed_single','mean_abs_Air_error_nm':agg['absolute_Air_error_nm']['mean'],'mean_film_MAE_nm':agg['film_MAE_nm']['mean'],'boundary_hit_rate':agg['boundary_hit_rate'],'mean_sigma_min_Jw':agg['sigma_min_Jw']['mean'],'mean_condition_number_Jw':agg['condition_number_Jw']['mean'],'mean_log10_det':agg['log10_det_JwT_Jw']['mean']})
 for band,r in by_band3.items():
  for mode,agg in r['aggregates'].items():rows.append({'stage':'stage3','band':band,'mode':mode,'mean_abs_Air_error_nm':agg['absolute_Air_error_nm']['mean'],'mean_film_MAE_nm':agg['film_MAE_nm']['mean'],'boundary_hit_rate':agg['boundary_hit_rate'],'mean_sigma_min_Jw':agg['sigma_min_Jw']['mean'],'mean_condition_number_Jw':agg['condition_number_Jw']['mean'],'mean_log10_det':agg['log10_det_JwT_Jw']['mean']})
 comparisons=[];noise_rows=[]
 for band,r in by_band3.items():
  case={(x['filename'],x['mode']):x for x in r['cases']};names=sorted({x['filename'] for x in r['cases']})
  for metric in ('absolute_Air_error_nm','film_MAE_nm','sigma_min_Jw','condition_number_Jw','log10_det_JwT_Jw'):
   d=np.asarray([case[(n,'joint_two_angle')][metric]-case[(n,'single_primary')][metric] for n in names]);higher=metric in {'sigma_min_Jw','log10_det_JwT_Jw'};wins=d>0 if higher else d<0
   comparisons.append({'comparison':'joint_minus_single','band':band,'metric':metric,'direction':'higher' if higher else 'lower','mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'joint_win_rate':float(np.mean(wins)),'paired_cases':len(d)})
  grouped=defaultdict(list)
  for x in r['cases']:grouped[(x['noise_case'],x['mode'])].append(x)
  for noise in sorted({k[0] for k in grouped}):
   for mode in ('single_primary','joint_two_angle'):
    g=grouped[(noise,mode)];noise_rows.append({'band':band,'noise_case':noise,'mode':mode,'realizations':len(g),'mean_abs_Air_error_nm':float(np.mean([x['absolute_Air_error_nm'] for x in g])),'mean_film_MAE_nm':float(np.mean([x['film_MAE_nm'] for x in g])),'boundary_hit_rate':float(np.mean([bool(x['boundary_hits']) for x in g])),'mean_sigma_min_Jw':float(np.mean([x['sigma_min_Jw'] for x in g])),'mean_condition_number_Jw':float(np.mean([x['condition_number_Jw'] for x in g]))})
 # Isolated multi-angle band comparison, same groups and same angle pair.
 a220={(x['filename'],x['mode']):x for x in by_band3['220-580']['cases']};a800={(x['filename'],x['mode']):x for x in by_band3['200-800']['cases']};names=sorted({x['filename'] for x in by_band3['220-580']['cases']})
 for metric in ('absolute_Air_error_nm','film_MAE_nm','sigma_min_Jw','condition_number_Jw','log10_det_JwT_Jw'):
  d=np.asarray([a800[(n,'joint_two_angle')][metric]-a220[(n,'joint_two_angle')][metric] for n in names]);higher=metric in {'sigma_min_Jw','log10_det_JwT_Jw'};wins=d>0 if higher else d<0;comparisons.append({'comparison':'joint_200_800_minus_joint_220_580','band':'paired','metric':metric,'direction':'higher' if higher else 'lower','mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'joint_win_rate':float(np.mean(wins)),'paired_cases':len(d)})
 write_csv(out/'v15_stage1_stage3_summary.csv',rows);write_csv(out/'v15_stage3_paired_comparisons.csv',comparisons);write_csv(out/'v15_stage3_noise_case_means.csv',noise_rows)
 report={'overall':'PASS','stage1_stage3_summary':rows,'paired_comparisons':comparisons,'source_model':'digitized EQ-99X','stage3_angle_pair_deg':by_band3['220-580']['cases'][0]['true_angles_deg'],'scope':'Typical only; each noise type is averaged over 10 realizations; no cross-noise-level averaging.'}
 (out/'v15_stage1_stage3_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 lookup={(x['stage'],x['band'],x['mode']):x for x in rows};s=lookup[('stage3','220-580','single_primary')];j=lookup[('stage3','220-580','joint_two_angle')];w=lookup[('stage3','200-800','joint_two_angle')]
 lines=['# V15 EQ-99X Stage 1 and Stage 3 analysis','', '- All statistics use typical noise only and preserve ten-realization means per noise type.', '- Stage 3 uses the same fixed measured-angle pair in both bands.', '', '| comparison | Air abs mean (nm) | film MAE mean (nm) | boundary rate | sigma_min(Jw) |','|---|---:|---:|---:|---:|',f'| 220-580 single primary | {s["mean_abs_Air_error_nm"]:.6g} | {s["mean_film_MAE_nm"]:.6g} | {s["boundary_hit_rate"]:.4f} | {s["mean_sigma_min_Jw"]:.6g} |',f'| 220-580 joint two-angle | {j["mean_abs_Air_error_nm"]:.6g} | {j["mean_film_MAE_nm"]:.6g} | {j["boundary_hit_rate"]:.4f} | {j["mean_sigma_min_Jw"]:.6g} |',f'| 200-800 joint two-angle | {w["mean_abs_Air_error_nm"]:.6g} | {w["mean_film_MAE_nm"]:.6g} | {w["boundary_hit_rate"]:.4f} | {w["mean_sigma_min_Jw"]:.6g} |']
 (out/'v15_stage1_stage3_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'overall':'PASS','output':str(out)},indent=2))
if __name__=='__main__':main()
