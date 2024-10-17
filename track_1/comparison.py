import json
import os
from models.utils import *

def find_string_in_file(document, string):
    '''
    This function performs a find as in Ctrl+f on a file.
    file_path: document already loaded as a dictionary.
    string: string to file in the file
    '''
    for key in document:
        if key == string:
            return True
    return False

def is_pred_correct(pred_label, document, string):
    is_malware = find_string_in_file(document, string)
    return (pred_label == 0 and not is_malware) or (pred_label == 1 and is_malware)

if __name__ == "__main__":

    submissions_path = os.path.join(os.path.dirname(__file__), "submissions/")
    comparison_path = os.path.join(os.path.dirname(__file__), "results/comparison.json")
    results_path = os.path.join(os.path.dirname(__file__), "results/results.json")

    base_path = os.path.join(os.path.dirname(__file__))
    names_proposed_path = os.path.join(base_path, "../data/names_proposed.json")
    types_proposed_path = os.path.join(base_path, "../data/types_proposed.json")

    with open(names_proposed_path, "r") as na:
        with open(types_proposed_path, "r") as ty:
            na_doc = json.load(na)
            ty_doc = json.load(ty)
            results = {}
            accuracies = {}
            avg_scores = {}
            results_acc_scores = {}
            count = 0
            for filename in os.listdir(submissions_path):
                if filename == ".gitkeep":
                    continue
                accuracies[filename.split(".")[0]] = 0
                avg_scores[filename.split(".")[0]] = 0
                with open(os.path.join(submissions_path, filename), 'r') as f:
                    print("________________________", filename)
                    list = json.load(f)
                    for test in list:
                        for sha256 in test:
                            if sha256 in results:
                                results[sha256] += test[sha256]
                            else:
                                results[sha256] = test[sha256]
                            accuracies[filename.split(".")[0]] += 1 \
                                    if is_pred_correct(test[sha256][0], na_doc, sha256) or is_pred_correct(test[sha256][0], ty_doc, sha256) else 0
                            avg_scores[filename.split(".")[0]] += test[sha256][1]
                            count += 1
                            print(f"count = {count}")
            for model in accuracies:
                results_acc_scores[model] = [accuracies[model] / count]
            for model in avg_scores:
                results_acc_scores[model] += [avg_scores[model] / count]

    with open(comparison_path, 'w') as f:
        json.dump(results, f)
    with open(results_path, "w") as f:
        json.dump(results_acc_scores, f)
    

