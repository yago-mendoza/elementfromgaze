"""The published weights: downloaded once from the GitHub release into models/, checked by SHA-256.

  generic   the base for new users: two eye networks trained on 10 strangers (MPIIFaceGaze), plus
            the strangers' eye patches every warm-up replays. CC BY-NC-SA 4.0, like the dataset.
  author    the author's own eye networks (two CNNs, 113k parameters each), trained on ~30 of his
            sessions. Only the networks: no images, no features, no sessions. For reproducing the
            numbers or as a different starting point. CC BY 4.0.

The files are Python pickles, and a pickle can run code when loaded: only load files whose hash
matches the one below (fetch() refuses anything else).
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RELEASE = "https://github.com/yago-mendoza/elementfromgaze/releases/download/weights-v1/"
FILES = {
    "generic": {"generic_base.pkl": "bf31aa65685d34d3ceb7964627a385ade4b03f71cb3a48fdaa9a256d1e5eec5b",
                "generic_replay.npz": "47a0909387b277f40d93332cc1b06a92dc32620085d76afe8caf906089e632f6"},
    "author": {"author_eyes.pkl": "9e8ad28171ba7d18c63de3984d4b23aa75e430bd4014617126bd473a4a4080d5"},
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(which: str = "generic", quiet: bool = False) -> bool:
    """Download (if missing) and verify one set. True when every file is present and intact."""
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES[which].items():
        path = MODELS / name
        if path.exists() and _sha(path) == sha:
            continue
        if not quiet:
            print(f"descargando {name} ({which})…")
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(RELEASE + name, tmp)
        except OSError as e:
            print(f"no se pudo descargar {name}: {e}")
            tmp.unlink(missing_ok=True)
            return False
        if _sha(tmp) != sha:
            tmp.unlink(missing_ok=True)
            print(f"{name}: el hash no coincide; no lo uso")
            return False
        tmp.replace(path)
    return True


def main(argv: list[str]) -> None:
    which = argv or ["generic"]
    for w in which:
        if w not in FILES:
            sys.exit(f"pesos desconocidos: {w} (hay: {', '.join(FILES)})")
        ok = fetch(w)
        print(f"{w}: {'listo en ' + str(MODELS) if ok else 'falló'}")
