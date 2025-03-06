import json
import os
import models
from models import DREBIN
from models.utils import *
import time
from sortedcontainers import SortedList
from feature_space_attack import FeatureSpaceAttack
import logging
import itertools

def main():
    LIM = 10
    good_features_tr = itertools.islice(load_samples_features(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"), 0), LIM*10)
    mal_features_tr = itertools.islice(load_samples_features(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"), 1), LIM)
    features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))
    #y_tr = load_labels(
    #    os.path.join(base_path, "../data/training_set_features.zip"),
    #    os.path.join(base_path, "../data/training_set.zip"))[:LIM]
    
    with open(os.path.join(data_base_path, "onlyMal100_feats.json")) as f:
        poss_manipulations = json.load(f)
        poss_manipulations.pop("TOTAL")
    for (key, values) in poss_manipulations.items():
        poss_manipulations[key] = SortedList(values)
    
    classifier, clf_path, vect_path = parse_model("FFNN_normal_small_CEL0109_")
    if os.path.exists(clf_path) and os.path.exists(vect_path):
        print("THERE ARE PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier = classifier.load(vect_path, clf_path)
        classifier.set_input_features(features_tr)
    
    attack = FeatureSpaceAttack(classifier=classifier, best_fitness_min_thresh=0.5,
                                logging_level=logging.INFO)
    attack.set_possible_manipulations(poss_manipulations)

    results = attack.run(mal_features_tr, good_features_tr, n_iterations=100,
                         n_features=30, n_candidates=50)
    with open(os.path.join(os.path.dirname(__file__), "test_results.txt"), "w") as f:
        f.write(str(results))


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "..")
    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    data_base_path = os.path.join(base_path, "../data")
    main()