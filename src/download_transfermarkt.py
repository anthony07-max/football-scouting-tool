"""
Download the CC0 Transfermarkt dataset by David Cariboo
(https://github.com/dcaribou/transfermarkt-datasets) and extract the CSV tables
into data/raw/tm/. The snapshot used for this project is current to July 2026.

Usage:  python src/download_transfermarkt.py
"""
import io
import os
import zipfile

import requests

URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "tm")
TABLES = {"players", "player_valuations", "appearances", "games", "clubs", "transfers", "competitions"}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Downloading Transfermarkt datasets (~240 MB) ...")
    r = requests.get(URL, timeout=600)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base.split(".")[0] in TABLES:
                with open(os.path.join(OUT, base), "wb") as f:
                    f.write(z.read(name))
                print("  saved", base)


if __name__ == "__main__":
    main()
