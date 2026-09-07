"""Verify frozen Phase 3 identifiability and scaled diagnostics."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from .backend.v10_source import source_sha256

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def main():
    project=Path(__file__).resolve().parent.parent; manifest=json.loads((project/'PHASE3_DIAGNOSTICS_FREEZE_MANIFEST.json').read_text(encoding='utf-8')); errors=[]
    if source_sha256()!=manifest['formal_v10_sha256']: errors.append('formal V10 source hash mismatch')
    if sha256_file(project/'PHASE3_FREEZE_MANIFEST.json')!=manifest['phase3_freeze_manifest_sha256']: errors.append('Phase 3 freeze manifest mismatch')
    for relative,expected in manifest['files'].items():
        path=project/relative
        if not path.is_file(): errors.append(f'missing: {relative}')
        elif sha256_file(path)!=expected: errors.append(f'hash mismatch: {relative}')
    report={'phase3_diagnostics_job_id':manifest['gpu_job_id'],'checked_files':len(manifest['files']),'pass':not errors,'errors':errors,'optimizer_scaling':manifest['optimizer_scaling'],'freeze_rule':manifest['freeze_rule']}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(2)
if __name__=='__main__': main()
