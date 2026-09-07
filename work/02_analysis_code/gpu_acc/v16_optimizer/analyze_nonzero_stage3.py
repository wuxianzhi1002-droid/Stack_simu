"""Compare V15 zero-angle and V16 optimized nonzero-angle Stage 3 results."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

METRICS=("absolute_Air_error_nm","film_MAE_nm","sigma_min_Jw","condition_number_Jw","log10_det_JwT_Jw")
HIGHER={"sigma_min_Jw","log10_det_JwT_Jw"}

def args():
 p=argparse.ArgumentParser();p.add_argument("--v15-220",type=Path,required=True);p.add_argument("--v15-800",type=Path,required=True);p.add_argument("--v16-220",type=Path,required=True);p.add_argument("--v16-800",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);return p.parse_args()
def load(p,version):
 r=json.loads(p.read_text(encoding="utf-8"));assert r["version"]==version and r["overall"]=="PASS";assert r["summary"]["processed_groups"]==100 and r["summary"]["fit_rows"]==200 and r["summary"]["closure_passed"]==200;return r
def key(r):return (r["noise_case"],int(r["realization_index"]),r["mode"])
def mean(a):return float(np.mean(np.asarray(list(a),dtype=float)))
def main():
 a=args();v15={"220-580":load(a.v15_220,"v15_stage3_multiangle_joint_gpu"),"200-800":load(a.v15_800,"v15_stage3_multiangle_joint_gpu")};v16={"220-580":load(a.v16_220,"v16_stage3_nonzero_multiangle_joint_gpu"),"200-800":load(a.v16_800,"v16_stage3_nonzero_multiangle_joint_gpu")}
 summary=[];noise=[];paired=[]
 for version,reports in (("V15_0.00_0.20",v15),("V16_0.02_0.20",v16)):
  for band,report in reports.items():
   for mode,agg in report["aggregates"].items():summary.append({"version":version,"band":band,"mode":mode,"mean_abs_Air_error_nm":agg["absolute_Air_error_nm"]["mean"],"mean_film_MAE_nm":agg["film_MAE_nm"]["mean"],"boundary_hit_rate":agg["boundary_hit_rate"],"mean_sigma_min_Jw":agg["sigma_min_Jw"]["mean"],"mean_condition_number_Jw":agg["condition_number_Jw"]["mean"],"mean_log10_det":agg["log10_det_JwT_Jw"]["mean"]})
   groups=defaultdict(list)
   for row in report["cases"]:groups[(row["noise_case"],row["mode"])].append(row)
   for (case,mode),rows in sorted(groups.items()):noise.append({"version":version,"band":band,"noise_case":case,"mode":mode,"realizations":len(rows),"mean_abs_Air_error_nm":mean(x["absolute_Air_error_nm"] for x in rows),"mean_film_MAE_nm":mean(x["film_MAE_nm"] for x in rows),"mean_sigma_min_Jw":mean(x["sigma_min_Jw"] for x in rows),"mean_condition_number_Jw":mean(x["condition_number_Jw"] for x in rows)})
 for band in v16:
  old={key(x):x for x in v15[band]["cases"]};new={key(x):x for x in v16[band]["cases"]};assert old.keys()==new.keys()
  for metric in METRICS:
   for mode in ("single_primary","joint_two_angle"):
    keys=sorted(k for k in old if k[2]==mode);d=np.asarray([new[k][metric]-old[k][metric] for k in keys]);wins=d>0 if metric in HIGHER else d<0
    paired.append({"comparison":"V16_nonzero_minus_V15_zero","band":band,"mode":mode,"metric":metric,"mean_delta":float(d.mean()),"median_delta":float(np.median(d)),"v16_win_rate":float(np.mean(wins)),"paired_cases":len(d)})
  current={key(x):x for x in v16[band]["cases"]}
  for metric in METRICS:
   ids=sorted({(k[0],k[1]) for k in current});d=np.asarray([current[(c,i,"joint_two_angle")][metric]-current[(c,i,"single_primary")][metric] for c,i in ids]);wins=d>0 if metric in HIGHER else d<0
   paired.append({"comparison":"V16_joint_minus_single","band":band,"mode":"paired","metric":metric,"mean_delta":float(d.mean()),"median_delta":float(np.median(d)),"v16_win_rate":float(np.mean(wins)),"paired_cases":len(d)})
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 def write(name,rows):
  with (out/name).open("w",encoding="utf-8-sig",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write("v16_stage3_summary.csv",summary);write("v16_stage3_noise_case_means.csv",noise);write("v16_stage3_paired_comparisons.csv",paired)
 payload={"version":"v16_nonzero_angle_stage3_analysis","created_utc":datetime.now(timezone.utc).isoformat(),"scope":"typical only; ten realizations retained per noise type; no pooling across noise types for case-level interpretation","summary":summary,"paired_comparisons":paired,"noise_case_means":noise}
 (out/"v16_nonzero_angle_stage3_analysis.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 get=lambda ver,band,mode:next(x for x in summary if x["version"]==ver and x["band"]==band and x["mode"]==mode)
 lines=["# V16 \u975e\u96f6\u89d2\u5ea6 Stage 3 \u5bf9\u6bd4\u5206\u6790","","## \u53e3\u5f84","","- V15: 0.00/0.20 deg theoretical boundary baseline.","- V16: enumerate 45 pairs on 0.02-0.20 deg grid and maximize robust worst-case sigma_min under 9 angle-error scenarios.","- EQ-99X, typical noise, 10 noise types x 10 realizations; preserve per-noise 10-realization means.","","## \u6c47\u603b","","| version | band | mode | Air abs mean (nm) | film MAE (nm) | sigma_min(Jw) | kappa(Jw) | log10 det |","|---|---|---|---:|---:|---:|---:|---:|"]
 for x in summary:lines.append(f"| {x['version']} | {x['band']} | {x['mode']} | {x['mean_abs_Air_error_nm']:.6g} | {x['mean_film_MAE_nm']:.6g} | {x['mean_sigma_min_Jw']:.6g} | {x['mean_condition_number_Jw']:.6g} | {x['mean_log10_det']:.6g} |")
 lines += ["","## V16 \u7ed3\u8bba",""]
 for band in ("220-580","200-800"):
  s=get("V16_0.02_0.20",band,"single_primary");j=get("V16_0.02_0.20",band,"joint_two_angle")
  air=(j["mean_abs_Air_error_nm"]/s["mean_abs_Air_error_nm"]-1)*100;sig=(j["mean_sigma_min_Jw"]/s["mean_sigma_min_Jw"]-1)*100;k=(j["mean_condition_number_Jw"]/s["mean_condition_number_Jw"]-1)*100
  lines.append(f"- {band} nm: \u8054\u5408\u540e sigma_min \u63d0\u9ad8 {sig:.2f}%, kappa \u53d8\u5316 {k:.2f}%, Air \u7edd\u5bf9\u8bef\u5dee\u53d8\u5316 {air:+.2f}%.")
 lines += ["","\u6392\u9664 0 deg \u540e\uff0c\u7b2c\u4e8c\u89d2\u5ea6\u4ecd\u63d0\u4f9b\u72ec\u7acb\u4fe1\u606f\u5e76\u6539\u5584\u6761\u4ef6\u6027\uff0c\u4f46\u5f53\u524d\u7b49\u6743\u8054\u5408\u6b8b\u5dee\u4e0b\u672a\u8f6c\u5316\u4e3a Air \u7cbe\u5ea6\u6536\u76ca\u30020.02 deg \u662f\u5019\u9009\u4e0b\u754c\uff0c\u82e5\u5b9e\u9a8c\u53ef\u9760\u4e0b\u754c\u66f4\u9ad8\uff0c\u5e94\u91cd\u65b0\u679a\u4e3e\u3002"]
 (out/"v16_nonzero_angle_stage3_analysis_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print(out)
if __name__=="__main__":main()
