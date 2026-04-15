"""
Download French mortality data from the Human Mortality Database (HMD).

Usage:
    python src/download_hmd_data.py

Prerequisites:
    - Free HMD account at https://www.mortality.org
    - Set environment variables HMD_USERNAME and HMD_PASSWORD,
      or enter credentials interactively.

Downloads:
    - Death rates (Mx) by sex, single age, single year
    - Population exposures
    - Saved to data/raw/
"""

import os
import sys
import getpass
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import base64

BASE_URL = "https://www.mortality.org/File/GetDocument/hmd.v6/FRA/STATS"

FILES = {
    "Mx_1x1.txt": "Death rates (both sexes combined)",
    "fMx_1x1.txt": "Death rates (females)",
    "mMx_1x1.txt": "Death rates (males)",
    "Exposures_1x1.txt": "Population exposures",
    "Deaths_1x1.txt": "Death counts",
    "Population.txt": "Population by age and sex",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_credentials() -> tuple[str, str]:
    username = os.environ.get("HMD_USERNAME")
    password = os.environ.get("HMD_PASSWORD")

    if not username:
        print("HMD credentials required (register free at https://www.mortality.org)")
        username = input("HMD username (email): ").strip()
    if not password:
        password = getpass.getpass("HMD password: ")

    return username, password


def download_file(filename: str, username: str, password: str) -> bool:
    url = f"{BASE_URL}/{filename}"
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()

    request = Request(url)
    request.add_header("Authorization", f"Basic {credentials}")

    dest = DATA_DIR / filename
    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
            dest.write_bytes(content)
        return True
    except HTTPError as e:
        print(f"  ERROR: HTTP {e.code} — {e.reason}")
        if e.code == 401:
            print("  Check your HMD credentials.")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HMD France Data Downloader")
    print("=" * 60)
    print(f"Target directory: {DATA_DIR}\n")

    username, password = get_credentials()
    print()

    success = 0
    for filename, description in FILES.items():
        print(f"Downloading {filename} ({description})...", end=" ")
        if download_file(filename, username, password):
            size_kb = (DATA_DIR / filename).stat().st_size / 1024
            print(f"OK ({size_kb:.0f} KB)")
            success += 1
        else:
            print("FAILED")

    print(f"\nDone: {success}/{len(FILES)} files downloaded to {DATA_DIR}")

    if success == 0:
        print("\nNo files downloaded. Verify your HMD credentials and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
