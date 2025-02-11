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
from sklearn.model_selection import train_test_split

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
    adv_X, _, adv_y, _ = train_test_split(adv_batch, labels, test_size=None, 
                                  train_size=n_good_samples + n_mal_samples-1,
                                  random_state=42)
    return (adv_X, adv_y)

def adversarial_training(X, y, classifier, attack, n_feats, step, ATsize, ATratio,
                         adv_examples_path):
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
        classifier._fit(X_sliced, y_sliced)
        adv_batch, adv_batch_labels = \
                generate_adv_batch(attack, ATsize, ATratio*ATsize, n_feats)
        with open(adv_examples_path, "a") as f:
            json.dump({i: adv_batch}, f, indent=2)
        # The vectorizer has already been fitted, still need to encode the samples
        classifier.fit(adv_batch, adv_batch_labels, fit=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-classifier",
                        help="The model on which to perform adversarial training")
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
    opt = parser.parse_args()
    print(f"Input arguments to the program:\n\
            classifier: {opt.classifier}\n\
            manipulation_algo : {opt.manipulation_algo}\n\
            manipulation_degree : {opt.manipulation_degree}\n\
            step: {opt.step}\n\
            ATsize : {opt.ATsize}\n\
            ATratio : {opt.ATratio}")

    model_string = f"AT_{opt.classifier}_{opt.manipulation_algo}" +\
                   f"_{opt.manipulation_degree}_{opt.step}_{opt.ATsize}_{opt.ATratio}"
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

    classifier, _, _ = parse_model(opt.classifier)
    
    if opt.manipulation_algo == "genetic":
        min_thresh = 0 if ("drebin" in opt.classifier or
                           "secsvm" in opt.classifier) else 0.5 
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
        adversarial_training(features_tr, y_tr, classifier, attack, opt.manipulation_degree,
                         opt.step, opt.ATsize, opt.ATratio, adv_examples_path)
        classifier.save(clf_path, vect_path)
    
    results = evaluate(classifier, min_thresh=0)

    with open(os.path.join(base_path, submission_path), "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__))
    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    main()