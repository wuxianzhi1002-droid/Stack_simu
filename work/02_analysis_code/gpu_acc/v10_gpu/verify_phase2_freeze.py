"""Verify that the formally passed Phase 2 baseline has not drifted."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def sha256_file(path: Path) -> str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""): d.update(block)
 return d.hexdigest()

def main() -> None:
 project=Path(__file__).resolve().parent.parent
 manifest=json.loads((project/"PHASE2_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
 errors=[]
 if sha256_file(project/"PHASE1_FREEZE_MANIFEST.json") != manifest["phase1_freeze_manifest_sha256"]: errors.append("Phase 1 freeze manifest hash mismatch")
 for rel,expected in manifest["files"].items():
  path=project/rel
  if not path.is_file(): errors.append(f"missing: {rel}")
  elif sha256_file(path) != expected: errors.append(f"hash mismatch: {rel}")
 report={"phase2_frozen_job_id":manifest["passed_job_id"],"checked_files":len(manifest["files"]),"pass":not errors,"errors":errors,"timing_contract":manifest["timing_contract"]}
 print(json.dumps(report,ensure_ascii=False,indent=2))
 if errors: raise SystemExit(2)
if __name__ == "__main__": main()
