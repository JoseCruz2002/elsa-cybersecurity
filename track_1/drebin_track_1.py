import json
import os
import models
from models import DREBIN
from models.utils import *
from evaluation import evaluate
import argparse


if __name__ == "__main__":
    """
    NB: in this example, the pre-extracted features are used. Alternatively,
    the APK file paths can be passed to the classifier.
    To fit the model, you can use `classifier.extract_features` to get the
    features and then pass them to `classifier.fit`.
    To classify the APK files, you can directly pass the list containing the
    file paths to `classifier.classify`.
    """
    parser = argparse.ArgumentParser()
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
    parser.add_argument("-adv_mode", choices=["genetic", "naive", ""], 
                        default="",
                        help="How the samples manipulation should be performed")
    parser.add_argument("-n_feats", default=5, type=int)
    parser.add_argument("-n_good_samples", default=67500, type=int)
    parser.add_argument("-n_mal_samples", default=7500, type=int)
    parser.add_argument("-sub_addition", default="",
                        help="When want to do a special submission")
    opt = parser.parse_args()
    print(f"Input arguments to the program:\n\
            feature_selection:  {opt.feat_selection}\n\
                param: {opt.param}\n\
                selection_type: {opt.selection_type}\n\
                selection_function: {opt.selection_function}\n\
                estimator: {opt.estimator}")

    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    base_path = os.path.join(os.path.dirname(__file__))

    feat_selection_args = {
        "feat_selection": opt.feat_selection,
        "param": opt.param,
        "selection_type": opt.selection_type,
        "selection_function": opt.selection_function,
        "estimator": opt.estimator
    }
    param_str = str(opt.param).replace('.', '') if opt.param < 1 else int(opt.param)
    model_variation = ""
    if opt.feat_selection != "":
        model_variation += f"_{opt.feat_selection}FS"
        if opt.feat_selection == "Variance":
            model_variation += f"-{param_str}"
        elif opt.feat_selection == "Univariate":
            model_variation += f"-{opt.selection_type}-{opt.selection_function}-{param_str}"
        elif opt.feat_selection in ("Recursive", "RecursiveCV"):
            model_variation += f"-{opt.estimator}-{param_str}"

    if opt.adv_mode != "":
        model_variation += \
            f"_adv-{opt.adv_mode}-Over-{opt.n_good_samples}-{opt.n_mal_samples}-{opt.n_feats}"  

    clf_path = os.path.join(
        model_base_path, f"pretrained/drebin{model_variation}_classifier.pth")
    vect_path = os.path.join(
        model_base_path, f"pretrained/drebin{model_variation}_vectorizer.pkl")
    submission_path = f"submissions/submission_drebin{model_variation}{opt.sub_addition}_track_1.json"
    
    print(f"clf_path: {clf_path}")
    print(f"vect_path: {vect_path}")
    print(f"submission_path: {submission_path}")
    classifier = DREBIN(C=0.1)

    if os.path.exists(clf_path) and os.path.exists(vect_path):
        print("THERE ARE PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier = DREBIN.load(vect_path, clf_path)
    else:
        print("THERE ARE NONONONONON PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))
        y_tr = load_labels(
            os.path.join(base_path, "../data/training_set_features.zip"),
            os.path.join(base_path, "../data/training_set.zip"))
        if opt.feat_selection != "":
            classifier.fit(features_tr, y_tr, feat_sel=True, args=feat_selection_args)
            with open(os.path.join(base_path, 
                                   f"selected_features/drebin{model_variation}.json"), "w") as f:
                json.dump(classifier.input_features, f, indent=2)
        else:
            classifier.fit(features_tr, y_tr)
        classifier.save(vect_path, clf_path)

    results = evaluate(classifier, min_thresh=0)

    with open(os.path.join(base_path, submission_path), "w") as f:
        json.dump(results, f)
