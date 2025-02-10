import argparse
import itertools

from feature_space_attack import FeatureSpaceAttack
from models.utils import *
import models
from models import FFNN, MyModel, DREBIN, SecSVM
import os
import logging
import json


def parse_model(classifier_str):
    '''
    Returns a tuple of three elements:
        The classifier
        The path to the classifier
        The path to the vectorizer
    '''
    print(f"parsing model {classifier_str}")

    base_path = os.path.join(os.path.dirname(__file__))
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
        hidden_size, layers = (10, 2) if "small" in classifier_str else (150, 3)
        classifier = FFNN(hidden_size=hidden_size, layers=layers)
    else:
        raise ValueError(f"Error: {classifier_str} does not exist!")
    return (classifier, clf_path, vect_path)


def perform_attack(classifier, attack, y_tr, num_feats_to_attack, num_samples_to_attack):
    base_path = os.path.join(os.path.dirname(__file__))
    goodware_features = (
        sample for sample, label in zip(load_features(
        os.path.join(base_path, "../data/training_set_features.zip")),
        y_tr) if label == 0)
    malware_features = itertools.islice(load_features(
        os.path.join(base_path, "../data/test_set_adv_features.zip")),
        num_samples_to_attack)
    adv_examples = attack.run(
        malware_features, goodware_features, n_iterations=100,
        n_features=num_feats_to_attack, n_candidates=50)
    y_pred, scores = classifier.predict(adv_examples)
    results = ({
        sha256: [int(y), float(s)] for sha256, y, s in zip(
            load_sha256_list(os.path.join(base_path,
                    f"../data/test_set_adv_features.zip"))[:num_samples_to_attack],
            y_pred, scores)})
    return results


def evaluate_acc(no_attack_acc, acc_treshold, new_results):
    acc = 0
    for (_, values) in new_results.items():
        pred = values[0]
        acc += 1 if pred == 1 else 0
    new_acc = acc / len(new_results)
    return ((no_attack_acc - new_acc) > acc_treshold, new_acc)


def get_model_acc(classifier_str):
    print("getting the model accuracy")
    results_path = os.path.join(os.path.dirname(__file__), "results/results.json")
    with open(results_path, "r") as f:
        saved_results = json.load(f)
    return saved_results[f"{classifier_str}_track_1"]["no_attack_malware_apks"]["Accuracy"]


def save_results(results, model_name):
    results_path = os.path.join(os.path.dirname(__file__), f"results/incr_eval_{model_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def main(model_choices):
    parser = argparse.ArgumentParser()
    parser.add_argument('-classifier', choices=model_choices, help="The model to evaluate")
    parser.add_argument('-num_feats_attacked_init', default=1, type=int,
                        help="The program will start the feature_space_attacks\
                            modifying these number of features")
    parser.add_argument('-num_feats_attacked_stop', default=1, type=int,
                        help="The program will stop the feature_space_attacks\
                            modifying these number of features")
    parser.add_argument('-num_feats_attacked_stride', default=1, type=int,
                        help="The increment in the number of features attacked")
    parser.add_argument('-acc_threshold', type=float, default=0.1,
                        help="""How much lost accuracy is too much""")
    parser.add_argument('-num_samples_to_attack', default=1250, type=int,
                        help="The number of malware samples to attack")
    opt = parser.parse_args()
    print(f"Input arguments to the program:\n\
            classifier: {opt.classifier}\n\
            num_feats_attacked_init: {opt.num_feats_attacked_init}\n\
            num_feats_attacked_stop: {opt.num_feats_attacked_stop}\n\
            num_feats_attacked_stride: {opt.num_feats_attacked_stride}\n\
            acc_threshold: {opt.acc_threshold}\n\
            num_samples_to_attack: {opt.num_samples_to_attack}")

    classifier, clf_path, vect_path = parse_model(opt.classifier)
    print(f"classifier path: {clf_path}")
    print(f"vectorizer path: {vect_path}")
    classifier = classifier.load(vect_path, clf_path)
    if "FFNN" in opt.classifier:
        features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))
        classifier.set_input_features(features_tr)

    no_attack_acc = get_model_acc(opt.classifier)
    print(f"The initial model accuracy is: {no_attack_acc}")

    results = {}
    min_thresh = 0 if opt.classifier in ("drebin", "secsvm") else 0.5
    attack = FeatureSpaceAttack(classifier=classifier, best_fitness_min_thresh=min_thresh,
                                logging_level=logging.INFO)
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))
    for n_feats in range(opt.num_feats_attacked_init, opt.num_feats_attacked_stop,
                         opt.num_feats_attacked_stride):
        attack_results = perform_attack(classifier, attack, y_tr, n_feats,
                                        opt.num_samples_to_attack)
        thresh_passed, new_acc = evaluate_acc(no_attack_acc, opt.acc_threshold, attack_results)
        print(f"The accuracy for {n_feats} is: {new_acc}")
        results[n_feats] = [new_acc, attack_results]
        if thresh_passed:
            save_results(results, opt.classifier)
            return
    save_results(results, opt.classifier)


if __name__ == "__main__":
    
    base_path = os.path.join(os.path.dirname(__file__))
    filenames = os.listdir(os.path.join(os.path.dirname(__file__),
                                        "../android-detectors/pretrained"))
    aux = list(filter(lambda x: x != ".gitkeep" and "vector" not in x, filenames))
    model_choices = list(map(lambda x: x[:-15], aux))

    main(model_choices)