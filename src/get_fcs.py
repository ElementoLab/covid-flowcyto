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


def main():
    request_type = "authenticate"
    creds = json.load(open(Path("~").expanduser() / ".cytobank.auth.json"))
    res1 = requests.post(API_URL + request_type, data=creds)
    res1.raise_for_status()
    headers = {"Authorization": "Bearer " + res1.json()["user"]["authToken"]}

    # # List experiments
    request_type = "experiments"
    experiments = ["309657", "309658", "309659", "309660"]

    for experiment in experiments:
        # #  List FCSs
        res2 = requests.get(
            API_URL + request_type + f"/{experiment}/" + "fcs_files", headers=headers
        )
        res2.raise_for_status()
        (OUTPUT_DIR / experiment).mkdir(exist_ok=True, parents=True)

        # # Download single FCS
        fcs = {x["id"]: x["filename"] for x in res2.json()["fcsFiles"]}
        for fid, fname in fcs.items():
            output_file = OUTPUT_DIR / experiment / fname.replace(" ", "_")

            if output_file.exists():
                continue
            print(experiment, fid)
            res3 = requests.get(
                API_URL + request_type + f"/{experiment}/" + f"fcs_files/{fid}/download",
                headers=headers,
            )
            with open(output_file, "wb") as f:
                f.write(res3.content)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
