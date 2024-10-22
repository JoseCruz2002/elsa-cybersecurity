import json
import os
import models
from models import FFNN
from models.utils import *
from evaluation import evaluate


if __name__ == "__main__":

    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    base_path = os.path.join(os.path.dirname(__file__))

    clf_path = os.path.join(
        model_base_path, "pretrained/FFNN_classifier.pth")
    vect_path = os.path.join(
        model_base_path, "pretrained/FFNN_vectorizer.pkl")
    
    features_tr = load_features(
            os.path.join(base_path, "../data/training_set_features.zip"))
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))
    
    classifier = FFNN(features_tr)
    
    if os.path.exists(clf_path) and os.path.exists(vect_path):
        print("THERE ARE PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier = classifier.load(vect_path, clf_path)
        classifier.set_input_features(features_tr)
    else:
        print("THERE ARE NONONONONON PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier.fit(features_tr, y_tr)
        classifier.save(vect_path, clf_path)

    results = evaluate(classifier)

    with open(os.path.join(
            base_path, "submissions/submission_FFNN_track_1.json"),
            "w") as f:
        json.dump(results, f)
    
    
    
    
    
    
    
    
    
    
    
    # testes meus
    #i = 0
    #for feature_tr in features_tr:
    #    if i % 7500 == 0:
    #        print(f"i = {i} -;-; feature_tr = {feature_tr}")
    #    i += 1
    #print(f"y_tr = {y_tr}; len(y_tr) = {len(y_tr)}")
    #