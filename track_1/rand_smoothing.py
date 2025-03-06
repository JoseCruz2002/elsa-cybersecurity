import os
import logging
import argparse
import ast
import itertools
import numpy
import json
from feature_space_attack import FeatureSpaceAttack
from evaluation import evaluate
import models
from models.utils import *
import torch
import random
import time

class Feature:
    def __init__(self, add, remove):
        self.add = add
        self.remove = remove


DREBIN_FEATURES = {
    "req_permissions": Feature(True, False),
    "activities": Feature(True, True),
    "services": Feature(True, True),
    "providers": Feature(True, True),
    "receivers": Feature(True, True),
    "features": Feature(True, False),
    "intent_filters": Feature(True, False),
    "used_permissions": Feature(True, False),
    "api_calls": Feature(True, True),
    "suspicious_calls": Feature(True, True),
    "urls": Feature(True, True)
}

RS_BATCH = 5000

def RS_sample_modification(sample, noise, input_features):
    if len(sample) == 0:
        print("Sample's length is zero.")
        return sample
    n_modifications = 0
    while n_modifications != noise:
        decision = random.randint(1, 2)
        if decision == 1:
            # Remove a feature
            idx = random.randint(0, len(sample)-1)
            feature = sample[idx]
            removable = DREBIN_FEATURES[feature.split("::")[0]].remove
            if removable:
                sample.remove(feature)
                n_modifications += 1
        else:
            # Add a feature
            idx = random.randint(0, len(input_features)-1)
            feature = input_features[idx]
            addable = DREBIN_FEATURES[feature.split("::")[0]].add
            if feature not in sample and addable:
                sample.append(feature)
                n_modifications += 1
    return sample

def perform_RS(classifier, X, y, noise, RS_samples_path):
    X_aux, X = itertools.tee(X, 2)
    classifier.vectorizer_fit(X, transform=False)
    X_aux = list(X_aux)
    for i in range(75000//RS_BATCH):
        X_batch = X_aux[i*RS_BATCH : (i+1)*RS_BATCH]
        y_batch = y[i*RS_BATCH : (i+1)*RS_BATCH]
        print(f"LENNNNNS: {len(X_batch)};;; {y_batch.shape}")
        time.sleep(1)
        for j in range(len(X_batch)):
            #print(f"j: {j}; sample: {X_batch[j]}")
            X_batch[j] = RS_sample_modification(X_batch[j], noise,
                                                classifier.input_features)
        #with open(RS_samples_path, "a") as f:
        #    json.dump(X_batch, f, indent=2)
        classifier.fit(X_batch, y_batch, fit=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-classifier",
                        help="The model on which to perform randomized smoothing")
    # Randomized Smoothing parameters
    parser.add_argument("-noise", type=int, default=2,
                        help="The number of features to change")
    # Feature Selection parameters
    parser.add_argument("-feat_selection", choices=["Variance", "Univariate", "Recursive",
                                                    "RecursiveCV", "SelectFromModel",
                                                    "Sequential", ""],
                        default="", type=str, required=False)
    parser.add_argument("-param", default=-1.0, type=float, required=False,
                        help="The parameter for feature selection")
    parser.add_argument("-selection_type", choices=['percentile', 'k_best', 'fpr',
                                                    'fdr', 'fwe', ''],
                        default="",
                        help="The type of selection for Univariate FS")
    parser.add_argument("-selection_function", choices=['chi2', 'mutual_info_classif',
                                                        'f_classif', ''],
                        default="",
                        help="The function used for Univariate FS")
    parser.add_argument("-estimator", default="",
                        help="The estimator used for Recursive FS")
    parser.add_argument("-direction", choices=["forward", "backward", ""],
                        default="", help="Direction of Sequential FS")
    opt = parser.parse_args()
    print(f"Input arguments to the program:\n\
            Classifier: {opt.classifier}\n\
            Randomized Smoothing parameters:\n\
                noise: {opt.noise}\n\
            Feature Selection parameters:\n\
                feature_selection: {opt.feat_selection}\n\
                param: {opt.param}\n\
                selection_type: {opt.selection_type}\n\
                selection_function: {opt.selection_function}\n\
                estimator: {opt.estimator}\n\
                direction: {opt.direction}")

    RS_string = f"RSv2_{opt.classifier}_{str(opt.noise).replace('.', '')}"
    param_str = str(opt.param).replace('.', '') if opt.param < 1 else int(opt.param)
    fs = ""
    if opt.feat_selection != "":
        fs += f"{opt.feat_selection}FS"
        if opt.feat_selection == "Variance":
            fs += f"-{param_str}"
        elif opt.feat_selection == "Univariate":
            fs += f"-{opt.selection_type}-{opt.selection_function}-{param_str}"
        elif opt.feat_selection in ("Recursive", "RecursiveCV"):
            fs += f"-{opt.estimator}-{param_str}"
        elif opt.feat_selection == "SelectFromModel":
            model_variation += f"-{opt.estimator}-{param_str}"
        elif opt.feat_selection == "Sequential":
            model_variation += f"-{opt.estimator}-{opt.direction}-{param_str}"
    model_string = f"{RS_string}_{fs}"

    RS_samples_path = os.path.join(base_path, f"RS_samples/{model_string}.json")
    clf_path = os.path.join(model_base_path, "pretrained/" +
                            model_string + "_classifier" +
                            (".pth" if "FFNN" in opt.classifier else ".pkl"))
    vect_path = os.path.join(model_base_path, "pretrained/" +
                             model_string + "_vectorizer.pkl")
    submission_path = f"submissions/submission_{model_string}_track_1.json"
    
    print(f"RS_sample_path:\n {RS_samples_path}")
    print(f"classifier_path:\n {clf_path}")
    print(f"vectorizer_path:\n {vect_path}")
    print(f"submission_path:\n {submission_path}")

    n_features = 1461078
    vocab = None
    if f"{fs}.json" in os.listdir(os.path.join(base_path, f"selected_features/")):
        with open(os.path.join(base_path, f"selected_features/{fs}.json"), "r") as f:
            vocab = json.load(f)
            n_features = len(vocab)

    classifier, _, _ = parse_model(opt.classifier, n_features, vocab, use_RS=True)
    
    min_thresh = 0 if ("drebin" in opt.classifier or
                       "secsvm" in opt.classifier) else 0.5
    
    if (os.path.exists(clf_path) and os.path.exists(vect_path)):
        print("THERE ARE PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier.load(clf_path, vect_path)
    else:
        features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))
        y_tr = load_labels(
            os.path.join(base_path, "../data/training_set_features.zip"),
            os.path.join(base_path, "../data/training_set.zip"))
        perform_RS(classifier, features_tr, y_tr, opt.noise, RS_samples_path)
        classifier.save(clf_path, vect_path)

    results = evaluate(classifier, min_thresh=min_thresh)

    with open(os.path.join(base_path, submission_path), "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__))
    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    main()