import json
import os
import models
from models import DREBIN
from models.utils import *

if __name__ == "__main__":
    """
    What is the reason for the 1.461.078 features?
    """
    base_path = os.path.join(os.path.dirname(__file__))
    features_tr = load_features(os.path.join(base_path, 
                                    "../data/training_set_features.zip"))
    res = {}
    i = 0
    for sample in features_tr:
        for feat in sample:
            key, value = feat.split("::")
            if key not in res:
                res[key] = [value]
            elif value not in res[key]:
                res[key] += [value]
        if i % 100 == 0:
            print(i)
        if i == 35000:
            break
        i += 1

    total = 0
    for key in res: 
        print(f"Number of different features for {key}: {len(res[key])}")
        total += len(res[key])
    print(f"Number of different features in TOTAL: {total}")
    
    with open(os.path.join(base_path, "../data/features.json"), "w") as f:
        json.dump(res, f, indent=4, sort_keys=True)
