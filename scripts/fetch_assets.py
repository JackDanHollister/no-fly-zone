"""Fetch third-party model assets (not committed: ~137 MB of meshes).

Currently: the DeepMind/Janelia flybody fruit fly model from MuJoCo Menagerie
(Apache-2.0) -> assets/flybody/. Run once per machine after cloning.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "assets" / "flybody"


def main() -> None:
    if (DEST / "fruitfly.xml").exists():
        print(f"already fetched: {DEST}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none",
             "--sparse", REPO, tmp],
            check=True,
        )
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", "flybody"],
                       check=True)
        DEST.parent.mkdir(exist_ok=True)
        shutil.copytree(Path(tmp) / "flybody", DEST)
    print(f"fetched flybody -> {DEST}")


if __name__ == "__main__":
    main()
