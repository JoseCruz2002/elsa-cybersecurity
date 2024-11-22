import json
import os
import argparse
import models
from models import FFNN
from models.utils import *
from evaluation import evaluate


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-training", choices=["normal", "ratioed", "only_mal"],
                        default="normal")
    parser.add_argument("-structure", choices=["small", "big"], default="big")
    parser.add_argument("-use_CEL", default=False, type=bool)
    parser.add_argument("-CEL_weight_pos_class", default=0.1, type=float)
    parser.add_argument("-CEL_weight_neg_class", default=0.9, type=float)
    parser.add_argument("-dense", default=False, type=bool)
    parser.add_argument("-sub_addition", default="",
                        help="When want to do a special submission")
    opt = parser.parse_args()
    print(f"Input arguments to the program:\n\
            training:  {opt.training}\n\
            structure: {opt.structure}\n\
            use_CEL:   {opt.use_CEL}\n\
                CEL_weight_pos_class: {opt.CEL_weight_pos_class}\n\
                CEL_weight_neg_class: {opt.CEL_weight_neg_class}\n\
            dense:     {opt.dense}")

    model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
    base_path = os.path.join(os.path.dirname(__file__))

    CEL_str = "CEL" + str(opt.CEL_weight_pos_class).replace(".", "") + \
                str(opt.CEL_weight_neg_class).replace(".", "") if opt.use_CEL else ""
    dense_str = "dense" if opt.dense else ""
    model_variation = f"_{opt.training}_{opt.structure}_{CEL_str}_{dense_str}"

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
    
    classifier = FFNN(training=opt.training, structure=opt.structure, use_CEL=opt.use_CEL,
                      CEL_weight_pos_class=opt.CEL_weight_pos_class, 
                      CEL_weight_neg_class=opt.CEL_weight_neg_class,
                      dense=opt.dense, features=features_tr)
    
    if os.path.exists(clf_path) and os.path.exists(vect_path):
        print("THERE ARE PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier = classifier.load(vect_path, clf_path)
        classifier.set_input_features(features_tr)
    else:
        print("THERE ARE NONONONONON PRETRAINED MODELS!!!!!!!!!!!!!!!!")
        classifier.fit(features_tr, y_tr)
        classifier.save(vect_path, clf_path)

    results = evaluate(classifier, min_thresh=0.5)

    with open(os.path.join(
            base_path, submission_path),
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