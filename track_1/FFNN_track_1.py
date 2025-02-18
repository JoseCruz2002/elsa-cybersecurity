import json
import os
import argparse
import models
from models import FFNN
from models.utils import *
from evaluation import evaluate


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # FFNN model hyperparameters
    parser.add_argument("-training", choices=["normal", "ratioed", "only_mal"],
                        default="normal")
    parser.add_argument("-structure", choices=["small", "big"], default="big")
    parser.add_argument("-use_CEL", default=False, type=bool)
    parser.add_argument("-CEL_weight_pos_class", default=0.1, type=float)
    parser.add_argument("-CEL_weight_neg_class", default=0.9, type=float)
    parser.add_argument("-dense", default=False, type=bool)
    # Feature Selection parameters
    parser.add_argument("-feat_selection", choices=["Variance", "Univariate", "Recursive",
                                                    "RecursiveCV", "L1-based",
                                                    "Tree-based", "Sequential", ""],
                        default="", type=str, required=True)
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
    # Data augmentation (after training) parameters
    parser.add_argument("-adv_mode", choices=["genetic", "naive", ""], 
                        default="",
                        help="How the samples manipulation should be performed")
    parser.add_argument("-n_feats", default=0, type=int)
    parser.add_argument("-n_good_samples", default=0, type=int)
    parser.add_argument("-n_mal_samples", default=0, type=int)
    # Adversarial Training parameters
    parser.add_argument("-manipulation_algo", choices=["genetic", "naive", ""], default="",
                        help="The algorithm used for the adversarial examples generation")
    parser.add_argument("-manipulation_degree", type=int, default=-1,
                        help="The degree of manipulation to create adversarial examples")
    parser.add_argument("-step", type=int, default=-1,
                        help="Amount of samples in normal training between each AT iteration")
    parser.add_argument("-ATsize", type=int, default=-1,
                        help="Size of AT iteration batch")
    parser.add_argument("-ATratio", type=int, default=-1,
                        help="Ratio of good to malware samples of an AT iteration batch")
    # Submission parameters
    parser.add_argument("-sub_addition", default="",
                        help="When want to do a special submission")
    opt = parser.parse_args()
    print(f"Input arguments to the program:\n\
            Model Hyperparameters:\n\
                training:  {opt.training}\n\
                structure: {opt.structure}\n\
                use_CEL:   {opt.use_CEL}\n\
                    CEL_weight_pos_class: {opt.CEL_weight_pos_class}\n\
                    CEL_weight_neg_class: {opt.CEL_weight_neg_class}\n\
                dense:     {opt.dense}\n\
            Feature Selection parameters:\n\
                feature_selection:  {opt.feat_selection}\n\
                param: {opt.param}\n\
                selection_type: {opt.selection_type}\n\
                selection_function: {opt.selection_function}\n\
                estimator: {opt.estimator}\n\
            Data augmentation (after training) parameters:\n\
                adv_mode: {opt.adv_mode}\n\
                n_feats: {opt.n_feats}\n\
                n_good_samples: {opt.n_good_samples}\n\
                n_mal_samples: {opt.n_mal_samples}\n\
            Adversarial Training parameters:
                manipulation_algo : {opt.manipulation_algo}\n\
                manipulation_degree : {opt.manipulation_degree}\n\
                step: {opt.step}\n\
                ATsize : {opt.ATsize}\n\
                ATratio : {opt.ATratio}\n\
            Submission parameters: {opt.sub_addition}")

    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    base_path = os.path.join(os.path.dirname(__file__))

    CEL_str = "CEL" + str(opt.CEL_weight_pos_class).replace(".", "") + \
                str(opt.CEL_weight_neg_class).replace(".", "") if opt.use_CEL else ""
    dense_str = "dense" if opt.dense else ""
    model_variation = f"_{opt.training}_{opt.structure}_{CEL_str}_{dense_str}"

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
    model_variation += f"_{fs}"

    if opt.adv_mode != "":
        model_variation += \
            f"_adv-{opt.adv_mode}-Over-{opt.n_good_samples}-{opt.n_mal_samples}-{opt.n_feats}"
    elif opt.manipulation_algo != "":
        model_variation = f"AT_{model_variation}_{opt.manipulation_algo}" +\
                          f"_{opt.manipulation_degree}_{opt.step}_{opt.ATsize}_{opt.ATratio}"

    clf_path = os.path.join(
        model_base_path, f"pretrained/FFNN{model_variation}_classifier.pth")
    vect_path = os.path.join(
        model_base_path, f"pretrained/FFNN{model_variation}_vectorizer.pkl")
    submission_path = f"submissions/submission_FFNN{model_variation}{opt.sub_addition}_track_1.json"
    
    print(f"clf_path: {clf_path}")
    print(f"vect_path: {vect_path}")
    print(f"submission_path: {submission_path}")
    
    features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))
    
    n_features = 1461078
    vocab = None
    if f"{fs}.json" in os.listdir(os.path.join(base_path, f"selected_features/")):
        with open(os.path.join(base_path, f"selected_features/{fs}.json"), "r") as f:
            vocab = json.load(f)
            n_features = len(vocab)

    classifier = FFNN(training=opt.training, structure=opt.structure, use_CEL=opt.use_CEL,
                      CEL_weight_pos_class=opt.CEL_weight_pos_class, 
                      CEL_weight_neg_class=opt.CEL_weight_neg_class,
                      dense=opt.dense, n_features=n_features, vocabulary=vocab)
    
    if os.path.exists(clf_path) and os.path.exists(vect_path):
        print("THERE ARE PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier = classifier.load(vect_path, clf_path)
        classifier.set_input_features(features_tr)
    else:
        print("THERE ARE NONONONONON PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier.fit(features_tr, y_tr)
        classifier.save(vect_path, clf_path)

    results = evaluate(classifier, min_thresh=0.5)

    with open(os.path.join(base_path, submission_path),"w") as f:
        json.dump(results, f)
