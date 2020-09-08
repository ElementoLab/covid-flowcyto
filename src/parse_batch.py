#!/usr/bin/env python

"""
This script finds the processing date of each sample from the FCS metadata.
"""

import struct

import pandas as pd  # type: ignore[import]
import flowkit as fk  # type: ignore[import]

from src.conf import *

panels = json.load(open(metadata_dir / "flow_variables2.json"))
metadata_file = metadata_dir / "annotation.pq"
meta = pd.read_parquet(metadata_file)

fcs_dir = data_dir / "fcs"

# Get processing date from FCS metadata
print("Getting metadata from FCS files.")
failures = list()
dates = {panel: pd.Series(dtype="object") for panel in panels}
for panel in panels:
    print(panel)

    for sample_id in meta["sample_id"].unique():
        if sample_id in dates[panel].index:
            print(f"Skipping panel {panel}, sample {sample_id}.")
            continue

        print(sample_id)
        sample_name = (
            meta.loc[
                meta["sample_id"] == sample_id, ["patient_code", "sample_id"]
            ]
            .drop_duplicates()
            .squeeze()
            .name
        )

        _id = int(sample_id.replace("S", ""))
        try:
            files = sorted(list(fcs_dir.glob(f"{_id}*{panel}*.fcs")))
            fcs_file = files[0]
            # ^^ this will get the most recent in case the are copies (*(1) files)
        except IndexError:
            try:
                fff = list(fcs_dir.glob(f"{_id}x{panel}*.fcs"))
                # assert len(fff) in [0, 1]
                fcs_file = fff[0]
            except IndexError:
                print(f"Sample {sample_id} is missing!")
                failures.append((panel, sample_id))
                continue

        try:
            s = fk.Sample(fcs_file)
            # this shouldn't happen anymore as correupt files aren't selected anymore
        except KeyboardInterrupt:
            raise
        except struct.error:
            print(f"Sample {sample_id} failed parsing FCS file!")
            failures.append((panel, sample_id))
            continue

        dates[panel][sample_id] = s.metadata["date"]


print("Concatenating and writing to disk.")
dates_df = pd.DataFrame(dates).apply(pd.to_datetime)
dates_df.index.name = "sample_id"
dates_df.to_csv(metadata_dir / "facs_dates.csv")

# reduce panels by ~mode (most common value)
max_dates = dates_df.apply(lambda x: x.value_counts().idxmax(), axis=1).rename(
    "processing_batch"
)
max_dates.to_csv(metadata_dir / "facs_dates.reduced.csv")

print("Finished.")
