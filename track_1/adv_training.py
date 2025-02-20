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

def generate_adv_batch(attack, n_mal_samples, n_good_samples, n_feats):
    """
    Generate n_mal_samples, that have been manipulated using the specified attack 
    """
    goodware_samples = load_samples_features(
                os.path.join(base_path, "../data/training_set_features.zip"),
                os.path.join(base_path, "../data/training_set.zip"), 0)
    malware_samples = load_samples_features(
                os.path.join(base_path, "../data/training_set_features.zip"),
                os.path.join(base_path, "../data/training_set.zip"), 1)

    adv_examples = attack.run(
        itertools.islice(malware_samples, n_mal_samples), 
        itertools.islice(goodware_samples, n_good_samples),
        n_iterations=100, n_features=n_feats, n_candidates=50)
    
    adv_batch = list(itertools.islice(goodware_samples, n_good_samples)) + adv_examples
    labels = numpy.concatenate((numpy.zeros(n_good_samples), numpy.ones(n_mal_samples)))
    # Randomize positions so that the good to malware ratio is maintained during training.
    indices = numpy.arange(len(adv_batch))
    numpy.random.shuffle(indices)
    adv_X = [adv_batch[i] for i in indices]
    adv_y = numpy.array([labels[i] for i in indices])

    return (adv_X, adv_y)

def adversarial_training(X, y, classifier, attack, n_feats, step, ATsize, ATratio,
                         adv_examples_path, do_RS, noise):
    """
    Perform the adversarial training.
    Arguments:
        X: unaltered input
        y: unaltered labels
        classifier: The model on which to perform adversarial training
        attack: The adversarial examples generator algortihm
        step: Amount of samples in normal training between each AT iteration
        ATsize: Size of AT iteration batch
        ATratio: Ratio of good to malware samples of an AT iteration batch
    """
    X = classifier.vectorizer_fit(X, transform=True)
    n_ATiterations = 75000 // step + (0 if 75000 % step == 0 else 1)
    print(f"n_ATiterations: {n_ATiterations}")
    for i in range(n_ATiterations):
        X_sliced = X[i*step : step*(i+1)]
        y_sliced = y[i*step : step*(i+1)]
        print(f"shape of X_sliced: {X_sliced.shape}")
        print(f"shape of y_sliced: {y_sliced.shape}")
        # No need to transform, X_sliced already is encoded to a binary vector
        classifier._fit(X_sliced, y_sliced, do_RS, noise)
        adv_batch, adv_batch_labels = \
                generate_adv_batch(attack, ATsize, ATratio*ATsize, n_feats)
        with open(adv_examples_path, "a") as f:
            json.dump({i: adv_batch}, f, indent=2)
        # The vectorizer has already been fitted, still need to encode the samples
        classifier.fit(adv_batch, adv_batch_labels, fit=False,
                       rand_smoothing=do_RS, noise=noise)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-classifier",
                        help="The model on which to perform adversarial training")
    # Adversarial Training parameters
    parser.add_argument("-manipulation_algo", choices=["genetic", "naive"],
                        help="The algorithm used for the adversarial examples generation")
    parser.add_argument("-manipulation_degree", type=int,
                        help="The degree of manipulation to create adversarial examples")
    parser.add_argument("-step", type=int,
                        help="Amount of samples in normal training between each AT iteration")
    parser.add_argument("-ATsize", type=int,
                        help="Size of AT iteration batch")
    parser.add_argument("-ATratio", type=int,
                        help="Ratio of good to malware samples of an AT iteration batch")
    # Randomized Smoothing parameters
    parser.add_argument("-noise", type=float, default=-1.0,
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
            Adversarial Training parameters:\n\
                manipulation_algo : {opt.manipulation_algo}\n\
                manipulation_degree : {opt.manipulation_degree}\n\
                step: {opt.step}\n\
                ATsize : {opt.ATsize}\n\
                ATratio : {opt.ATratio}\n\
            Randomized Smoothing parameters:\n\
                noise: {opt.noise}\n\
            Feature Selection parameters:\n\
                feature_selection: {opt.feat_selection}\n\
                param: {opt.param}\n\
                selection_type: {opt.selection_type}\n\
                selection_function: {opt.selection_function}\n\
                estimator: {opt.estimator}\n\
                direction: {opt.direction}")

    AT_string = f"AT_{opt.classifier}_{opt.manipulation_algo}" +\
                   f"_{opt.manipulation_degree}_{opt.step}_{opt.ATsize}_{opt.ATratio}"
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
    RS_string = ""
    do_RS = False
    if opt.noise > 0.0:
        RS_string = f"RS_{str(opt.noise).replace('.', '')}"
        do_RS = True
    model_string = f"{AT_string}_{RS_string}_{fs}"
    

    adv_examples_path = os.path.join(base_path, f"adv_examples_for_AT/{model_string}.json")
    clf_path = os.path.join(model_base_path, "pretrained/" +
                            model_string + "_classifier" +
                            (".pth" if "FFNN" in opt.classifier else ".pkl"))
    vect_path = os.path.join(model_base_path, "pretrained/" +
                             model_string + "_vectorizer.pkl")
    submission_path = f"submissions/submission_{model_string}_track_1.json"
    
    print(f"classifier_path:\n{clf_path}")
    print(f"vectorizer_path:\n{vect_path}")
    print(f"adv_examples_path:\n{adv_examples_path}")
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
    
    if opt.manipulation_algo == "genetic":
        attack = FeatureSpaceAttack(classifier=classifier,
                                    best_fitness_min_thresh=min_thresh,
                                    logging_level=logging.INFO)

    if (os.path.exists(clf_path) and os.path.exists(vect_path)):
        print("THERE ARE PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier.load(clf_path, vect_path)
    else:
        print("THERE ARE NONONONONON PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))
        y_tr = load_labels(
            os.path.join(base_path, "../data/training_set_features.zip"),
            os.path.join(base_path, "../data/training_set.zip"))
        adversarial_training(features_tr, y_tr, classifier, attack,
                             opt.manipulation_degree, opt.step, opt.ATsize,
                             opt.ATratio, adv_examples_path, do_RS, opt.noise)
        classifier.save(clf_path, vect_path)
    
    results = evaluate(classifier, min_thresh=min_thresh)

    with open(os.path.join(base_path, submission_path), "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__))
    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    main()