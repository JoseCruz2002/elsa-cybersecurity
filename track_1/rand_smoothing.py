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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-classifier",
                        help="The model on which to perform adversarial training")
    # Randomized Smoothing parameters
    parser.add_argument("-noise", type=float, default=0.25,
                        help="The standart deviation of the gaussian noise")
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

    RS_string = f"RS_{opt.classifier}_{str(opt.noise).replace('.', '')}"
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

    clf_path = os.path.join(model_base_path, "pretrained/" +
                            model_string + "_classifier" +
                            (".pth" if "FFNN" in opt.classifier else ".pkl"))
    vect_path = os.path.join(model_base_path, "pretrained/" +
                             model_string + "_vectorizer.pkl")
    submission_path = f"submissions/submission_{model_string}_track_1.json"
    
    print(f"classifier_path:\n{clf_path}")
    print(f"vectorizer_path:\n{vect_path}")
    print(f"submission_path:\n{submission_path}")

    n_features = 1461078
    vocab = None
    if f"{fs}.json" in os.listdir(os.path.join(base_path, f"selected_features/")):
        with open(os.path.join(base_path, f"selected_features/{fs}.json"), "r") as f:
            vocab = json.load(f)
            n_features = len(vocab)

    classifier, _, _ = parse_model(opt.classifier, n_features, vocab)
    
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
        classifier.fit(features_tr, y_tr, rand_smoothing=True, noise=opt.noise)
        classifier.save(clf_path, vect_path)

    results = evaluate(classifier, min_thresh=min_thresh)

    with open(os.path.join(base_path, submission_path), "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__))
    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    main()