from datasets import load_dataset
import json
import polars as pl


# pl.read_json("outputs/xlam60k2swahali/output.json")

# with open("outputs/xlam60k2swahali_v2/pawa-function-calling-60k.json", "r") as f:
#     data = json.load(f)
#
# print(len(data))


ds = load_dataset("json", data_dir="outputs/xlam60k2swahali")


print(ds["train"].features)
