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

def generate_adv_samples(attack, adv_mode, n_mal_samples, n_good_samples, n_feats,
                         adv_samples_path):
 
    goodware_samples = load_samples_features(
                os.path.join(base_path, "../data/training_set_features.zip"),
                os.path.join(base_path, "../data/training_set.zip"), 0)
    malware_samples = load_samples_features(
                os.path.join(base_path, "../data/training_set_features.zip"),
                os.path.join(base_path, "../data/training_set.zip"), 1)

    if adv_mode == "genetic":
        adv_examples = attack.run(
            itertools.islice(malware_samples, n_mal_samples), 
            itertools.islice(goodware_samples, n_good_samples),
            n_iterations=100, n_features=n_feats, n_candidates=50)

    with open(adv_samples_path, "w") as f:
        samples = list(itertools.islice(goodware_samples, n_good_samples)) + adv_examples
        f.write(str(samples))
    
    return samples


def data_augmentation_over_existing_model(classifier, adv_samples, adv_mode, n_good_samples, 
                                     n_mal_samples, n_feats):

    labels = numpy.concatenate((numpy.zeros(n_good_samples), numpy.ones(n_mal_samples)))
    indices = numpy.arange(len(labels))
    numpy.random.shuffle(indices)
    X, y = adv_samples[indices], labels[indices]
    classifier.fit(X, y, fit=False)
    
    aux_name = classifier.toString() + f"_adv-{adv_mode}-Over-{n_good_samples}-{n_mal_samples}-{n_feats}"
    new_vect_path = os.path.join(base_path, f"../android-detectors/pretrained/{aux_name}_vectorizer.pkl")
    new_clf_path = os.path.join(base_path, f"../android-detectors/pretrained/{aux_name}_classifier.pth")
    classifier.save(new_vect_path, new_clf_path)


def adv_training_from_zero(classifier, clf_path, vect_path, attack, X, y):
    pass


def main(model_choices: list[str]):

    parser = argparse.ArgumentParser()
    parser.add_argument("-classifier", choices=model_choices,
                        help="The model name to perform data augmentation over")
    parser.add_argument("-adv_mode", choices=["genetic", "naive"], 
                        default="genetic",
                        help="How the samples manipulation should be performed")
    parser.add_argument("-n_feats", default=5, type=int)
    parser.add_argument("-n_good_samples", default=67500, type=int)
    parser.add_argument("-n_mal_samples", default=7500, type=int)
    opt = parser.parse_args()
    print(f"Input arguments to the program:\n\
            classifier: {opt.classifier}\n\
            adv_mode: {opt.adv_mode}\n\
            n_feats: {opt.n_feats}\n\
            n_good_samples: {opt.n_good_samples}\n\
            n_mal_samples: {opt.n_mal_samples}")
    
    features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))

    classifier, clf_path, vect_path = parse_model(opt.classifier)
    classifier.load(vect_path, clf_path)
    if "FFNN" in opt.classifier:
        classifier.set_input_features(features_tr)

    aux = f"adv_samples_{opt.n_good_samples}-{opt.n_mal_samples}_{opt.n_feats}.txt"
    adv_samples_path = os.path.join(base_path, aux)

    min_thresh = 0 if ("drebin" in opt.classifier or "secsvm" in opt.classifier) else 0.5

    # Adversarial Samples generation
    if os.path.exists(adv_samples_path):
        print(f"Adversarial samples already exist - {adv_samples_path}")
        with open(adv_samples_path, "r") as f:
            aux = f.read()
        adv_samples = ast.literal_eval(aux)
    else:
        if opt.adv_mode == "genetic":
            attack = FeatureSpaceAttack(classifier=classifier,
                                        best_fitness_min_thresh=min_thresh,
                                        logging_level=logging.INFO)
        else:
            print(f"This adversarial mode: {opt.adv_mode} is not yet implemented!")
            return
        print(f"Generating adversarial samples to save in - {adv_samples_path}")
        adv_samples = generate_adv_samples(attack, opt.adv_mode, opt.n_mal_samples,
                                           opt.n_good_samples, opt.n_feats,
                                           adv_samples_path)
    # Training extension
    if os.path.exists(clf_path) and os.path.exists(vect_path):
        print(f"Performing adversarial training on existing model {opt.classifier}")
        data_augmentation_over_existing_model(classifier, adv_samples, opt.adv_mode,
                                         opt.n_good_samples, opt.n_mal_samples,
                                         opt.n_feats)
    else:
         raise ValueError ("Data Augmentation requires existing model!")

if __name__ == "__main__":

    base_path = os.path.join(os.path.dirname(__file__))
    filenames = os.listdir(os.path.join(os.path.dirname(__file__),
                                        "../android-detectors/pretrained"))
    aux = list(filter(lambda x: x != ".gitkeep" and "vector" not in x, filenames))
    model_choices = list(map(lambda x: x[:-15], aux))

    main(model_choices)