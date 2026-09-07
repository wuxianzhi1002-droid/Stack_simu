"""Generate paired V15 EQ-99X two-angle StackRT datasets for Stage 3."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from v13_optimizer.build_multiframe_detector_dataset import make_frame

MASTER_START_NM=200.0; MASTER_STOP_NM=800.0; MASTER_SAMPLING_NM=0.02; MASTER_POINTS=30001
ANGLE_SIGMA_DEG=0.001


def sha256(path:Path)->str:
    d=hashlib.sha256()
    with path.open('rb') as h:
        for b in iter(lambda:h.read(1024*1024),b''): d.update(b)
    return d.hexdigest()


def load_main_v15():
    source=Path(__file__).resolve().parents[3]/'01_simulation_models'/'01_Lumerical_Workflow'/'main_v15.py'
    spec=importlib.util.spec_from_file_location('_v15_stage3_main_v15_generator',source)
    if spec is None or spec.loader is None: raise ImportError(source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module,source


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--angle-design',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=10);p.add_argument('--seed',type=int,default=20260831)
    return p.parse_args()


def main():
    args=parse_args()
    if args.repeats!=10: raise ValueError('formal Stage 3 requires 10 realizations per noise case')
    design=json.loads(args.angle_design.read_text(encoding='utf-8'))
    if design.get('overall')!='PASS' or design.get('version')!='v15_stage3_two_angle_design': raise ValueError('invalid angle design')
    angles=np.asarray(design['production_pair_deg'],dtype=np.float64)
    if angles.shape!=(2,) or not (0.0<=angles[0]<angles[1]<=0.2): raise ValueError(f'invalid pair {angles}')
    main_v15,source_path=load_main_v15();main_v15.configure_v15()
    override=argparse.Namespace(wavelength_start_nm=MASTER_START_NM,wavelength_stop_nm=MASTER_STOP_NM,output_sampling_nm=MASTER_SAMPLING_NM)
    main_v15.apply_wavelength_cli_overrides(override);main_v15.configure_v15();kernel=main_v15._kernel
    cases=[f'{factor}_typical' for factor in kernel.NOISE_FACTORS]
    complete_index={name:i for i,name in enumerate(kernel.all_case_names())}
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    generator=main_v15.V15DatasetGenerator('api',output)
    original_realize=kernel.realize_noise; current_angle={'value':float(angles[0])}
    def fixed_realize(case_name,reported,internal,rng):
        metadata,components=original_realize(case_name,reported,internal,rng)
        theta=float(current_angle['value']);nominal=float(main_v15.V15_CONFIG['REFLECTOR_ANGLE_NOMINAL_DEG'])
        metadata.update({'reflector_angle_deg':theta,'reflector_angle_setpoint_deg':theta,
            'reflector_angle_deviation_from_nominal_deg':theta-nominal,'reflector_angle_error_deg':theta-nominal})
        return metadata,components
    kernel.realize_noise=fixed_realize
    rows=[];failures=[]
    try:
      with kernel.OpticalSolver('api',output/'_stackrt_batch_bridge') as solver:
       for case_name in cases:
        case_index=complete_index[case_name]
        for realization in range(args.repeats):
         base_seed=int(args.seed+case_index*1_000_000+realization); angle_data=[]; angle_metadata=[];frames=[];detector_audits=[];measured=[];detector_seeds=[]
         for angle_index,theta in enumerate(angles):
          current_angle['value']=float(theta)
          data,metadata=generator.generate_one(solver,case_name,realization,base_seed)
          detector_seed=int(202609050000+case_index*100000+realization*100+angle_index)
          frame,sample,reference,dark,audit=make_frame(data['expected_sample_electrons'],data['expected_reference_electrons'],data['pixel_response_error_rel'],float(metadata['detector_noise_multiplier']),main_v15.V15_CONFIG['SPECTROMETER'],detector_seed)
          angle_seed=int(202609060000+case_index*100000+realization*100+angle_index)
          measured_angle=float(theta+np.random.default_rng(angle_seed).normal(0.0,ANGLE_SIGMA_DEG))
          frames.append(frame);detector_audits.append(audit);measured.append(measured_angle);detector_seeds.append(detector_seed);angle_data.append(data);angle_metadata.append(metadata)
         if not np.array_equal(angle_data[0]['components']['true_pixel_center_wavelengths_nm'],angle_data[1]['components']['true_pixel_center_wavelengths_nm']): raise RuntimeError('systematic wavelength state changed across angles')
         if not np.array_equal(angle_data[0]['components']['source_power_relative_curve_internal'],angle_data[1]['components']['source_power_relative_curve_internal']): raise RuntimeError('source-noise realization changed across angles')
         first=angle_data[0];meta=angle_metadata[0];reported=np.asarray(first['components']['reported_wavelengths_nm']);layers=generator.layers
         payload={
          'wavelengths':reported/1000.0,'reported_wavelengths_nm':reported,'spectra_measured':np.asarray(frames,dtype=np.float64),'spectrum_measured':np.asarray(frames[0],dtype=np.float64),
          'true_reflector_angles_deg':angles,'measured_reflector_angles_deg':np.asarray(measured),'angle_measurement_errors_deg':np.asarray(measured)-angles,
          'true_reflector_angle_deg':np.asarray(angles[0]),'measured_reflector_angle_deg':np.asarray(measured[0]),'angle_measurement_sigma_deg':np.asarray(ANGLE_SIGMA_DEG),
          'angle_measurement_mode':np.asarray('fixed_independent_measurement_per_angle'),'angle_fixed_in_inversion':np.asarray(True),'reported_axis_is_fixed':np.asarray(True),
          'layer_names':np.asarray(kernel.LAYER_NAMES,dtype='U32'),'layer_thickness_um':np.asarray([layers[n] for n in kernel.LAYER_NAMES]),'true_air_um':np.asarray(layers['Air']),
          'noise_case':np.asarray(case_name),'noise_factor':np.asarray(meta['noise_factor']),'noise_level':np.asarray(meta['noise_level']),'realization_index':np.asarray(realization),'random_seed':np.asarray(base_seed),
          'internal_wavelength_margin_nm':np.asarray(generator.internal_margin_nm),'internal_wavelength_step_nm':np.asarray(main_v15.V15_CONFIG['INTERNAL_WAVELENGTH_STEP_NM']),'output_sampling_nm':np.asarray(generator.output_sampling_nm),
          'config_json':np.asarray(json.dumps(main_v15.V15_CONFIG,ensure_ascii=False,sort_keys=True)),'noise_realization_json':np.asarray(json.dumps(meta,ensure_ascii=False,sort_keys=True)),
          'generator_version':np.asarray('main_v15_multiangle'),'optical_backend':np.asarray('api'),'multiangle_provenance':np.asarray('two-angle local Lumerical stackrt with shared systematic realization and independently resampled detector/angle measurement noise'),
          'angle_design_sha256':np.asarray(sha256(args.angle_design)),'eq99x_source_csv_sha256':np.asarray(main_v15.V15_CONFIG['EQ99X_SOURCE_CSV_SHA256']),
          'detector_seeds':np.asarray(detector_seeds,dtype=np.int64),'detector_audit_json':np.asarray(json.dumps(detector_audits,sort_keys=True)),
          'material_n_real_rel_delta':np.asarray([meta['material_n_real_rel_delta'][n] for n in kernel.PERTURBED_MATERIALS]),'material_k_rel_delta':np.asarray([meta['material_k_rel_delta'][n] for n in kernel.PERTURBED_MATERIALS]),
         }
         target=output/f'multiangle_{case_name}_r{realization:04d}_seed{base_seed}.npz';np.savez_compressed(target,**payload)
         rows.append({'noise_case':case_name,'noise_factor':meta['noise_factor'],'noise_level':meta['noise_level'],'realization_index':realization,'random_seed':base_seed,'theta1_true_deg':angles[0],'theta2_true_deg':angles[1],'theta1_measured_deg':measured[0],'theta2_measured_deg':measured[1],'angle1_error_deg':measured[0]-angles[0],'angle2_error_deg':measured[1]-angles[1],'npz':target.name,'sha256':sha256(target)})
         print(f'[{len(rows)}/100] {target.name} angles={angles.tolist()}',flush=True)
    except Exception as exc:
      failures.append(f'{type(exc).__name__}: {exc}');raise
    finally: kernel.realize_noise=original_realize
    with (output/'dataset_index.csv').open('w',encoding='utf-8-sig',newline='') as h:
      w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest={'version':'v15_stage3_multiangle_stackrt','created_utc':datetime.now(timezone.utc).isoformat(),'dataset_count':len(rows),'noise_scope':'10 typical noise types x 10 realizations','master_band_nm':[200.0,800.0],'sampling_nm':0.02,'wavelength_points':30001,'true_angles_deg':angles.tolist(),'angle_measurement_sigma_deg':ANGLE_SIGMA_DEG,'shared_across_angles':'material, source-shape, wavelength-axis and PRNU systematic realization','independent_across_angles':'shot/read/dark detector resampling and angle measurement error','source_model':'digitized EQ-99X source_shape_peak_normalized','eq99x_source_csv_sha256':main_v15.V15_CONFIG['EQ99X_SOURCE_CSV_SHA256'],'angle_design':str(args.angle_design.resolve()),'angle_design_sha256':sha256(args.angle_design),'main_v15':str(source_path),'main_v15_sha256':sha256(source_path),'failures':failures}
    (output/'simulation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'overall':'PASS','count':len(rows),'angles_deg':angles.tolist(),'output':str(output)},indent=2),flush=True)

if __name__=='__main__':main()
