import json
import os
import models
from models import SecSVM
from models.utils import *
from evaluation import evaluate


if __name__ == "__main__":
    """
    NB: in this example, the pre-extracted features are used. Alternatively,
    the APK file paths can be passed to the classifier.
    To fit the model, you can use `classifier.extract_features` to get the
    features and then pass them to `classifier.fit`.
    To classify the APK files, you can directly pass the list containing the
    file paths to `classifier.classify`.
    """
    classifier = SecSVM(C=0.1, lb=-0.5, ub=0.5)

    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    base_path = os.path.join(os.path.dirname(__file__))

    clf_path = os.path.join(
        model_base_path, "pretrained/secsvm_classifier.pkl")
    vect_path = os.path.join(
        model_base_path, "pretrained/secsvm_vectorizer.pkl")

    if os.path.exists(clf_path) and os.path.exists(vect_path):
        print("THERE ARE PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier = SecSVM.load(vect_path, clf_path)
    else:
        print("THERE ARE NONONONONON PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))
        y_tr = load_labels(
            os.path.join(base_path, "../data/training_set_features.zip"),
            os.path.join(base_path, "../data/training_set.zip"))
        classifier.fit(features_tr, y_tr)
        classifier.save(vect_path, clf_path)

    results = evaluate(classifier, min_thresh=0)


    with open(os.path.join(
            base_path, "submissions/submission_secsvm_track_1.json"),
            "w") as f:
        json.dump(results, f)
