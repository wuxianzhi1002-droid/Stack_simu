"""Verify the frozen Phase 5 implementation and deterministic 10-NPZ result."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    root=Path(__file__).resolve().parents[1];manifest=json.loads((root/"PHASE5_FREEZE_MANIFEST.json").read_text(encoding="utf-8"));errors=[]
    for rel,expected in manifest["files"].items():
        path=(root/rel).resolve()
        if not path.is_file():errors.append(f"missing: {rel}")
        elif digest(path)!=expected:errors.append(f"hash mismatch: {rel}")
    result=json.loads((root/manifest["result_json"]).read_text(encoding="utf-8"))
    if not result.get("acceptance",{}).get("pass"):errors.append("Phase 5 result is not PASS")
    if len(result.get("cases",[]))!=10:errors.append("Phase 5 case count is not 10")
    if result.get("stop_condition")!="Stopped after deterministic 10-NPZ Phase 5 run; 401-NPZ dataset was not launched.":errors.append("stop condition changed")
    out={"phase5_frozen_job_id":manifest["job_id"],"checked_files":len(manifest["files"]),"pass":not errors,"errors":errors,"cache_contract":manifest["cache_contract"],"stop_rule":manifest["stop_rule"]}
    print(json.dumps(out,indent=2));raise SystemExit(0 if not errors else 2)
if __name__=="__main__":main()
