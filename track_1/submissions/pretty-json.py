import os
import json

path = "/home/josecruz/Documents/MEIC/Thesis/elsa-cybersecurity/track_1/submissions/submission_FFNN_dropout_track_1.json"

with open(path, "r") as f:
    j = json.load(f)

pretty_path = path.split(".")[0] + "_pretty." + path.split(".")[1]
with open(pretty_path, "w") as nf:
    json.dump(j, nf, indent=2)