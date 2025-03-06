import json
import os
import models
from models.utils import *
import time
from sortedcontainers import SortedList

def complete_features_test():
    """
    What is the reason for the 1.461.078 features?
    """
    features_tr = load_features(os.path.join(base_path, 
                                    "../data/training_set_features.zip"))
    res = {}
    i = 0
    for sample in features_tr:
        for feat in sample:
            key, value = feat.split("::")
            if key not in res:
                res[key] = SortedList([value])
            elif value not in res[key]:
                res[key].add(value)
        if i % 100 == 0:
            print(i)
        i += 1

    res["TOTAL"] = dict(list((cat, 0) for cat in res))
    for key in res:
        if key != "TOTAL":
            res[key] = list(res[key])
            res["TOTAL"][key] = len(res[key])
    
    with open(os.path.join(base_path, "../data/features.json"), "w") as f:
        json.dump(res, f, indent=4, sort_keys=True)


def good_vs_malware_features():
    """
    What are the features that only appear on goodware and the ones that only
    appear on malware?
    """
    good_features_tr = list(load_samples_features(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"),
        0))
    mal_features_tr = list(load_samples_features(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"),
        1))
    b = time.time()
    goodware_features = {}
    malware_features = {}
    i, j = 0, 0
    for sample in mal_features_tr:
        time_begin = time.time()
        for feat in sample:
            key, value = feat.split("::")
            if key not in malware_features:
                malware_features[key] = SortedList([value])
            elif value not in malware_features[key]:
                malware_features[key].add(value)
        if i % 100 == 0:
            print(f"i: {i}; time: {time.time() - time_begin}")
        i += 1
    for sample in good_features_tr:
        time_begin = time.time()
        for feat in sample:
            key, value = feat.split("::")
            if value not in malware_features[key]:
                if key not in goodware_features:
                    goodware_features[key] = SortedList([value])
                elif value not in goodware_features[key]:
                    goodware_features[key].add(value)
        if j % 100 == 0:
            print(f"j: {j}; time: {time.time() - time_begin}")
        j += 1
    time_end = time.time()
    print(f"Total time elapsed: {time_end - b}")

    for (key, values_list) in malware_features.items():
        if key in goodware_features:
            for value in values_list:
                if value in goodware_features[key]:
                    malware_features[key].remove(value)
    
    malware_features["TOTAL"] = dict(list((cat, 0) for cat in malware_features))
    goodware_features["TOTAL"] = dict(list((cat, 0) for cat in goodware_features))
    for key in malware_features:
        if key != "TOTAL":
            malware_features[key] = list(malware_features[key])
            malware_features["TOTAL"][key] = len(malware_features[key])
    for key in goodware_features:
        if key != "TOTAL":
            goodware_features[key] = list(goodware_features[key])
            goodware_features["TOTAL"][key] = len(goodware_features[key])

    with open(os.path.join(base_path, "../data/onlyGood_feats.json"), "w") as f:
        json.dump(goodware_features, f, indent=4, sort_keys=True)
    with open(os.path.join(base_path, "../data/onlyMal_feats.json"), "w") as f:
        json.dump(malware_features, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__))
    complete_features_test()
    good_vs_malware_features()
