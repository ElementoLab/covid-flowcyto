import json

import pandas as pd
import flowkit as fk

from src.conf import *

panels = json.load(open(metadata_dir / "flow_variables.json"))
metadata_file = metadata_dir / "annotation.pq"
meta = pd.read_parquet(metadata_file)

# Extract matrix, gate

dates = dict()
for panel in panels:
    print(panel)

    _dates = dict()
    for sample_id in meta["sample_id"].unique():
        print(sample_id)
        sample_name = (
            meta.loc[meta["sample_id"] == sample_id, ["patient_code", "sample_id"]]
            .drop_duplicates()
            .squeeze()
            .name
        )

        fcs_dir = data_dir / "fcs" / panels[panel]["num"]
        # TODO: check for more files
        _id = int(sample_id.replace("S", ""))
        try:
            fcs_file = list(fcs_dir.glob(f"{_id}_" + panels[panel]["id"] + "*.fcs"))[0]
        except IndexError:
            try:
                fff = list(fcs_dir.glob(f"{_id}x" + "*.fcs"))
                # assert len(fff) in [0, 1]
                fcs_file = fff[0]
            except IndexError:
                print(f"Sample {sample_id} is missing!")
                continue

        s = fk.Sample(fcs_file)

        _dates[sample_id] = s.metadata["date"]

    dates[panel] = pd.Series(_dates)


dates = pd.concat({k: pd.to_datetime(v) for k, v in dates.items()}, 1)
dates.index.name = "sample_id"
dates.to_csv(metadata_dir / "facs_dates.csv")

# reduce panels by ~mode (most common value)
max_dates = dates.apply(lambda x: x.value_counts().idxmax(), axis=1).rename("processing_batch")
max_dates.to_csv(metadata_dir / "facs_dates.reduced.csv")
