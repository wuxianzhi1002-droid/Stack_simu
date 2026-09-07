"""Verify that the formally passed Phase 1 baseline has not drifted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    project_dir=Path(__file__).resolve().parent.parent
    manifest_path=project_dir/"PHASE1_FREEZE_MANIFEST.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    errors=[]
    for relative,expected in manifest["files"].items():
        path=project_dir/relative
        if not path.is_file(): errors.append(f"missing: {relative}")
        elif sha256_file(path) != expected: errors.append(f"hash mismatch: {relative}")
    report={"phase1_frozen_job_id":manifest["passed_job_id"],"checked_files":len(manifest["files"]),"pass":not errors,"errors":errors,"timing_audit":manifest["timing_audit"]}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(2)


if __name__ == "__main__": main()
