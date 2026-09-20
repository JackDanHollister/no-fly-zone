"""A provenance stamp for every artefact this project writes.

Every number in the README traces to a run; the run should be traceable
to the code and the brain that produced it. stamp() records the git
commit (and whether the tree was dirty), the time and host, the package
versions that matter, the script's arguments and the brain's own metadata
(dataset, corrections, wiring control), and a short id hashed from the
parts that determine the result (commit, brain, arguments), so two
artefacts with the same id came from the same code and settings.

dump_json() adds it under "provenance" to a dict payload (a list payload
gets a sidecar <name>.provenance.json so its schema stays what it was);
savez() adds it to an .npz as a JSON string; attach() adds it to a dict
bound for torch.save.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_PACKAGES = ("numpy", "scipy", "pandas", "mujoco", "ale_py", "gymnasium", "torch", "triton")


def git_state(root: Path = ROOT) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = run("rev-parse", "HEAD")
    return {"commit": commit or None, "short": commit[:10] or None,
            "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")) if commit else None}


def package_versions() -> dict:
    out = {}
    for name in _PACKAGES:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    return out


def brain_meta(brain) -> dict | None:
    if brain is None:
        return None
    meta = dict(getattr(brain, "meta", {}) or {})
    for k in ("version", "backend", "n"):
        v = getattr(brain, k, None)
        if v is not None and k not in meta:
            meta[k] = v
    try:
        from .brain_backend import describe
        meta["describe"] = describe(brain)
    except Exception:
        pass
    return meta


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


def stamp(args=None, brain=None, **extra) -> dict:
    a = vars(args) if args is not None and not isinstance(args, dict) else args
    a = {k: _jsonable(v) for k, v in (a or {}).items()}
    git = git_state()
    bm = brain_meta(brain)
    core = json.dumps({"commit": git["commit"], "brain": bm, "args": a}, sort_keys=True, default=str)
    out = {"id": hashlib.sha256(core.encode()).hexdigest()[:12],
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
           "python": sys.version.split()[0], "git": git, "packages": package_versions(),
           "brain": bm, "args": a}
    out.update({k: _jsonable(v) for k, v in extra.items()})
    return out


def dump_json(path, payload, args=None, brain=None, **extra) -> Path:
    path = Path(path)
    prov = stamp(args=args, brain=brain, **extra)
    if isinstance(payload, dict):
        payload = {**payload, "provenance": prov}
        path.write_text(json.dumps(payload, indent=1, default=float))
    else:
        path.write_text(json.dumps(payload, indent=1, default=float))
        path.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))
    return path


def savez(path, args=None, brain=None, **arrays) -> Path:
    import numpy as np
    np.savez(path, provenance=json.dumps(stamp(args=args, brain=brain)), **arrays)
    return Path(path)


def attach(payload: dict, args=None, brain=None, **extra) -> dict:
    return {**payload, "provenance": stamp(args=args, brain=brain, **extra)}
