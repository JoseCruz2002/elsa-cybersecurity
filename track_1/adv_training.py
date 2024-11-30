import os
import logging
import argparse
import itertools
import numpy

from feature_space_attack import FeatureSpaceAttack
import models
from models.utils import *
from models import FFNN, MyModel, DREBIN, SecSVM


def parse_model(classifier_str: str):
    '''
    Returns a tuple of three elements:
        The classifier
        The path to the classifier
        The path to the vectorizer
    '''
    print(f"parsing model {classifier_str}")

    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    file_extension = "pth" if "FFNN" in classifier_str else "pkl"
    clf_path = os.path.join(model_base_path, f"pretrained/{classifier_str}_classifier.{file_extension}")
    vect_path = os.path.join(model_base_path, f"pretrained/{classifier_str}_vectorizer.pkl")

    if classifier_str == "MyModel":
        classifier = MyModel()
    elif classifier_str == "secsvm":
        classifier = SecSVM(C=0.1, lb=-0.5, ub=0.5)
    elif classifier_str == "drebin":
        classifier = DREBIN(C=0.1)
    elif "FFNN" in classifier_str:
        aux = classifier_str.split("_")
        training = aux[1]
        structure = aux[2]
        cel = True if "CEL" in classifier_str else False
        cel_pos_class = float(aux[3][3:5])/10 if cel else 0
        cel_neg_class = float(aux[3][5:])/10 if cel else 0
        dense = True if "dense" in classifier_str else False
        classifier = FFNN(training=training, structure=structure, use_CEL=cel,
                      CEL_weight_pos_class=cel_pos_class, 
                      CEL_weight_neg_class=cel_neg_class,
                      dense=dense, features=[])
    else:
        print(f"Error: {classifier_str} does not exist!")
    return (classifier, clf_path, vect_path)


def adv_training_over_existing_model(classifier, clf_path, vect_path, X, attack,
                                     adv_mode, n_feats, n_good_samples, n_mal_samples):

    classifier.load(vect_path, clf_path)
    classifier.set_input_features(features=X)

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
    
    adv_ex_file = f"adv_examples_{n_good_samples}-{n_mal_samples}_{n_feats}.txt"
    with open(os.path.join(base_path, adv_ex_file), "w") as f:
        f.write(str(adv_examples))

    classifier.fit(adv_examples, numpy.ones(len(adv_examples)))
    
    aux_name = classifier.toString() + f"_adv-{adv_mode}-Over-{n_good_samples}-{n_mal_samples}-{n_feats}"
    new_vect_path = os.path.join(base_path, f"../android_detectors/pretrained/{aux_name}_vectorizer.pkl")
    new_clf_path = os.path.join(base_path, f"../android_detectors/pretrained/{aux_name}_classifier.pth")
    classifier.save(new_vect_path, new_clf_path)


def adv_training_from_zero(classifier, clf_path, vect_path, attack, X, y):
    pass


def main(model_choices: list[str]):

    parser = argparse.ArgumentParser()
    parser.add_argument("-classifier", choices=model_choices,
                        help="The model name to perform adversarial training")
    parser.add_argument("-adv_mode", choices=["genetic", "naive"], 
                        default="genetic",
                        help="How the samples manipulation should be performed")
    parser.add_argument("-n_feats", default=5, type=int)
    parser.add_argument("-n_good_samples", default=67500, type=int)
    parser.add_argument("-n_mal_samples", default=7500, type=int)
    opt = parser.parse_args()
    print(f"Input arguments to the program:\n\
            classifier:  {opt.classifier}\n\
            adv_mode: {opt.adv_mode}\n\
            n_feats: {opt.n_feats}\n\
            n_good_samples: {opt.n_good_samples}\n\
            n_mal_samples: {opt.n_mal_samples}")
    
    classifier, clf_path, vect_path = parse_model(opt.classifier)

    features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))

    if opt.adv_mode == "genetic":
        min_thresh = 0 if opt.classifier in ("drebin", "secsvm") else 0.5
        attack = FeatureSpaceAttack(classifier=classifier, best_fitness_min_thresh=min_thresh,
                                    logging_level=logging.INFO)
    else:
        print(f"This adversarial mode: {opt.adv_mode} is not yet implemented!")
        return

    if os.path.exists(clf_path) and os.path.exists(vect_path):
        print(f"Performing adversarial training on existing model {opt.classifier}")
        adv_training_over_existing_model(classifier, clf_path, vect_path, X=features_tr,
                                         attack=attack, adv_mode=opt.adv_mode,
                                         n_feats=opt.n_feats, 
                                         n_good_samples=opt.n_good_samples,
                                         n_mal_samples=opt.n_mal_samples)
    else:
        print(f"Performing adversarial training on new model {opt.classifier}")
        adv_training_from_zero(classifier, clf_path, vect_path, attack,
                               features_tr, y_tr)


if __name__ == "__main__":

    base_path = os.path.join(os.path.dirname(__file__))
    filenames = os.listdir(os.path.join(os.path.dirname(__file__),
                                        "../android-detectors/pretrained"))
    aux = list(filter(lambda x: x != ".gitkeep" and "vector" not in x, filenames))
    model_choices = list(map(lambda x: x[:-15], aux))

    main(model_choices)