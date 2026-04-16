"""Download French death-rate data from the Human Mortality Database (HMD).

Connects to the HMD website and downloads files containing death rates,
population counts, and death counts for France. The data is saved locally
so we can use it for analysis without downloading it again.

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

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
#
# 1. Asks for your HMD login (from environment variables or typed in)
# 2. Connects to the HMD website and downloads 6 data files for France
# 3. Saves them to the data/raw/ folder so other scripts can use them
# ───────────────────────────────────────────────────────

import base64
import getpass
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# The web address where HMD stores the French mortality data files
BASE_URL = "https://www.mortality.org/File/GetDocument/hmd.v6/FRA/STATS"

# Each file we need to download, and a short description of what it contains
FILES = {
    "Mx_1x1.txt": "Death rates (both sexes combined)",
    "fMx_1x1.txt": "Death rates (females)",
    "mMx_1x1.txt": "Death rates (males)",
    "Exposures_1x1.txt": "Population exposures",
    "Deaths_1x1.txt": "Death counts",
    "Population.txt": "Population by age and sex",
}

# Where to save the downloaded files (data/raw/ folder in the project)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_credentials() -> tuple[str, str]:
    """Get the username and password needed to access HMD.

    First checks if they're stored in environment variables (which is
    the easiest way if you download data often). If not found, asks
    you to type them in.

    Returns:
        A pair of (username, password) strings.
    """
    # Try to get credentials from environment variables first
    username = os.environ.get("HMD_USERNAME")
    password = os.environ.get("HMD_PASSWORD")

    # If not set, ask the user to type them in
    if not username:
        print("HMD credentials required (register free at https://www.mortality.org)")
        username = input("HMD username (email): ").strip()
    if not password:
        password = getpass.getpass("HMD password: ")

    return username, password


def download_file(filename: str, username: str, password: str) -> bool:
    """Download one file from the HMD website.

    Builds the download link, attaches your login credentials, and
    saves the file to the data folder. If anything goes wrong (bad
    password, network error, etc.), it prints an error message.

    Args:
        filename: The name of the file to download (e.g., "Mx_1x1.txt").
        username: Your HMD email address.
        password: Your HMD password.

    Returns:
        True if the file was downloaded successfully, False if something
        went wrong.
    """
    # Build the full download URL
    url = f"{BASE_URL}/{filename}"

    # Encode the login credentials the way the web server expects
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()

    # Create the download request with login info attached
    request = Request(url)
    request.add_header("Authorization", f"Basic {credentials}")

    # Try to download and save the file
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
    """Run the full download process from start to finish.

    Creates the output folder if needed, gets your login credentials,
    then downloads all the data files one by one. Tells you how many
    succeeded at the end, and exits with an error if none worked.
    """
    # Make sure the output folder exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HMD France Data Downloader")
    print("=" * 60)
    print(f"Target directory: {DATA_DIR}\n")

    # Get login credentials
    username, password = get_credentials()
    print()

    # Download each file and keep track of how many succeed
    success = 0
    for filename, description in FILES.items():
        print(f"Downloading {filename} ({description})...", end=" ")
        if download_file(filename, username, password):
            size_kb = (DATA_DIR / filename).stat().st_size / 1024
            print(f"OK ({size_kb:.0f} KB)")
            success += 1
        else:
            print("FAILED")

    # Print a summary of what happened
    print(f"\nDone: {success}/{len(FILES)} files downloaded to {DATA_DIR}")

    # If nothing downloaded at all, exit with an error
    if success == 0:
        print("\nNo files downloaded. Verify your HMD credentials and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
