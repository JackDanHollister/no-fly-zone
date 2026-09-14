"""Fetch third-party assets and data (gitignored; ~570 MB total).

Run once per machine after cloning:
- flybody (MuJoCo Menagerie, Apache-2.0): the fly body model, ~137 MB
- Shiu et al. brain model (MIT): FlyWire v783 connectivity + reference
  implementation, ~380 MB
- flywire_annotations (Jefferis lab): cell types + soma positions, ~51 MB
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _clone(repo: str, dest: Path, sparse_dir: str | None = None) -> None:
    if dest.exists():
        print(f"already fetched: {dest}")
        return
    dest.parent.mkdir(exist_ok=True)
    if sparse_dir is None:
        subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)],
                       check=True)
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        "--sparse", repo, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", sparse_dir],
                       check=True)
        shutil.copytree(Path(tmp) / sparse_dir, dest)
    print(f"fetched {dest}")


def main() -> None:
    _clone("https://github.com/google-deepmind/mujoco_menagerie.git",
           ROOT / "assets" / "flybody", sparse_dir="flybody")
    _clone("https://github.com/philshiu/Drosophila_brain_model",
           ROOT / "data" / "shiu_model")
    _clone("https://github.com/flyconnectome/flywire_annotations",
           ROOT / "data" / "flywire_annotations")
    print("all assets ready")


if __name__ == "__main__":
    main()
