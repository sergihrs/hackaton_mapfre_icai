"""
Descarga el dataset COIL 2000 (TICDATA2000.txt y TICEVAL2000.txt) desde UCI
y lo guarda en data/. Ejecutar desde la raíz del proyecto:

    python scripts/download_coil2000.py
"""

import urllib.request
from pathlib import Path

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILES = [
    ("ticdata2000.txt", "TICDATA2000.txt"),
    ("ticeval2000.txt", "TICEVAL2000.txt"),
]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url_name, local_name in FILES:
        path = DATA_DIR / local_name
        url = f"{BASE}/{url_name}"
        print(f"Descargando {url} -> {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"  OK ({path.stat().st_size} bytes)")
    print("Listo. Ficheros en", DATA_DIR)


if __name__ == "__main__":
    main()
