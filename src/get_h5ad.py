import json
import requests

from src.conf import metadata_dir, results_dir, data_dir


urls = json.load(open(metadata_dir / "h5ad_urls.json"))

ssp = results_dir / "single_cell"
ssp.mkdir()

for t in urls:
    if t == "raw":
        ending = ".concatenated.full.h5ad"
    elif t == "processed":
        ending = ".concatenated.full.processed.h5ad"

    for name, url in urls[t].items():
        file = ssp / name / name + ending
        if not file.exists():
            print(file)
            r = requests.get(url)
            with open(file, "wb") as f:
                f.write(r.content)

output_dir = data_dir / "h5ad"
output_dir.mkdir()
with open(output_dir / "__done__", "w"):
    pass
