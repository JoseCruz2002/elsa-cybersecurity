import json
import os
from models.utils import *
import numpy as np
from sklearn import metrics as metr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

TEST_1 = "no_attack_goodware_apks"
TEST_2 = "no_attack_malware_apks"
TEST_3 = "feature_space_attack_2"
TEST_4 = "feature_space_attack_5"
TEST_5 = "feature_space_attack_10"

def submission_tests():
    submission_path = os.path.join(os.path.dirname(__file__), "submissions/submission_drebin_track_1.json")
    with open(submission_path, "r") as f:
        sub = json.load(f)
        print(f"len(sub) = {len(sub)}")
        print(f"type of sub: {type(sub)}")

def comparison_tests():
    comparison_path = os.path.join(os.path.dirname(__file__), "results/comparison.json")
    with open(comparison_path, "r") as f:
        comp = json.load(f)
        print(f"len(comp) = {len(comp)}")


def join_all_submissions():

    test_names = {0: TEST_1, # no attack during the test in the goodware test set
                  1: TEST_2, # no attack during the test in the malware test set
                  2: TEST_3, # attacked and modified 2 features
                  3: TEST_4, # attacked and modified 5 features
                  4: TEST_5 # attacked and modified 10 features
                  }

    submissions_path = os.path.join(os.path.dirname(__file__), "submissions/")

    all_subs = {}
    for filename in os.listdir(submissions_path):
        if filename == ".gitkeep" or "pretty" in filename:
            continue
        model_name = filename.split(".")[0]
        all_subs[model_name] = {}
        with open(os.path.join(submissions_path, filename), 'r') as f:
            print("________________________", filename)
            list = json.load(f)
            i = 0
            for test in list: # test is a dictionary with as many keys as shas
                all_subs[model_name][test_names[i]] = {}
                aux = all_subs[model_name][test_names[i]]
                for sha256 in test:
                    if sha256 in aux:
                        aux[sha256] += test[sha256]
                    else:
                        aux[sha256] = test[sha256]
                i += 1
    
    #with open(os.path.join(os.path.dirname(__file__), "results/comparison.json"), "w") as f:
    #    json.dump(all_subs, f, indent=2)

    return all_subs


def prediction_correct(pred_class, test):

    mal = [TEST_2, TEST_3, TEST_4, TEST_5]
    good = [TEST_1]

    return (pred_class == 0 and test in good) or \
           (pred_class == 1 and test in mal)

def calculate_metrics(all_subs:dict):

    results_path = os.path.join(os.path.dirname(__file__), "results/results.json")
    results = {}

    for model in all_subs: # the first key of all_subs is a model (drevin, FFNN,...)
        results[model] = {}
        for test in all_subs[model]: # each model has 5 tests, see test_names in above function for more details
            results[model][test] = {
                "TP": 0,
                "FP": 0,
                "TN": 0,
                "FN": 0,
                "Accuracy": 0,
                "Score": 0
            }
            metrics = results[model][test]
            for sha256 in all_subs[model][test]:
                pred_class = all_subs[model][test][sha256][0]
                if pred_class == 1:
                    if prediction_correct(pred_class, test):
                        metrics["TP"] += 1
                    else:
                        metrics["FP"] += 1
                else:
                    if prediction_correct(pred_class, test):
                        metrics["TN"] += 1
                    else:
                        metrics["FN"] += 1
                metrics["Accuracy"] += 1 if prediction_correct(pred_class, test) else 0
                metrics["Score"] += all_subs[model][test][sha256][1]
            metrics["Accuracy"] /= len(all_subs[model][test])
            metrics["Score"] /= len(all_subs[model][test])

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def create_confusion_matrices(metrics):

    results_dir_path = os.path.join(os.path.dirname(__file__), "results/")

    for model in metrics:
        for test in metrics[model]:
            TP = metrics[model][test]["TP"]
            FP = metrics[model][test]["FP"]
            TN = metrics[model][test]["TN"]
            FN = metrics[model][test]["FN"]
            confusion_matrix = np.array([[TP, FP], [FN, TN]])
            confusion_df = pd.DataFrame(confusion_matrix, 
                            columns=["Actual Malware", "Actual Goodware"], 
                            index=["Predicted Malware", "Predicted Goodware"])
            # Plot the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues", 
                        cbar=False, annot_kws={"size": 16}, linewidths=1, 
                        linecolor='black', square=True)
            plt.title(f"Confusion Matrix for {test} for {model} ", fontsize=18)
            plt.xlabel("Predicted Label", fontsize=14)
            plt.ylabel("Actual Label", fontsize=14)

            plt.savefig(f"{results_dir_path}cm_{test}_{model}.png", dpi=300, bbox_inches='tight')
            plt.close()



if __name__ == "__main__":
    #submission_tests()
    #comparison_tests()
    all_subs = join_all_submissions()
    metrics = calculate_metrics(all_subs)
    create_confusion_matrices(metrics)
