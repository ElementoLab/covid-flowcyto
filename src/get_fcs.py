#!/usr/bin/env python


"""
Download all FCS files from Cytobank.
"""


import sys
import json
from pathlib import Path

import requests


API_URL = "https://premium.cytobank.org/cytobank/api/v1/"
OUTPUT_DIR = Path("data") / "fcs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
EXPERIMENTS = ["308893"]


def main():
    request_type = "authenticate"
    creds = json.load(open(Path("~").expanduser() / ".cytobank.auth.json"))
    res1 = requests.post(API_URL + request_type, data=creds)
    res1.raise_for_status()
    headers = {"Authorization": "Bearer " + res1.json()["user"]["authToken"]}

    # # List experiments
    request_type = "experiments"
    # res12 = requests.get(
    #     API_URL + request_type, headers=headers)
    # res12.raise_for_status()
    # res12.json()['experiments']
    # EXPERIMENTS = ["309657", "309658", "309659", "309660"]

    for experiment in EXPERIMENTS:
        # #  List FCSs
        res2 = requests.get(
            API_URL + request_type + f"/{experiment}/" + "fcs_files", headers=headers
        )
        res2.raise_for_status()

        # # Download single FCS
        fcs = {x["id"]: x["filename"] for x in res2.json()["fcsFiles"]}
        for fid, fname in fcs.items():
            output_file = OUTPUT_DIR / fname.replace(" ", "_")

            if output_file.exists():
                print(f"{output_file} already exists, skipping...")
                continue
            print(experiment, fid)
            res3 = requests.get(
                API_URL + request_type + f"/{experiment}/" + f"fcs_files/{fid}/download",
                headers=headers,
            )
            with open(output_file, "wb") as f:
                f.write(res3.content)

    open(OUTPUT_DIR / "__done__", "w")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        quit()
    else:
        quit()
    finally:
        quit()
