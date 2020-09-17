import json
from pathlib import Path
from glob import glob

import boto3
from tqdm import tqdm


def hook(t):
    """
    Wraps tqdm instance. Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    """
    last_b = [0]

    def inner(b=1, tsize=None, bsize=1):
        """
        b  : int, optional
            Number of blocks just transferred [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


NAME = "covid-flowcyto"

auth_file = Path("~/.wasabi.auth.json").expanduser()
auth = json.load(open(auth_file))

s3 = boto3.resource("s3", endpoint_url="https://s3.wasabisys.com", **auth)
buck = s3.create_bucket(Bucket=NAME)

h5ads = Path("results/single_cell").glob("*/*full*.h5ad")

for file in h5ads:
    print(file)
    with tqdm(unit="B", unit_scale=True) as t:  # all optional kwargs
        buck.upload_file(
            file.as_posix(),
            "h5ad/" + file.name,
            ExtraArgs={"ACL": "public-read"},
            Callback=hook(t),
        )
