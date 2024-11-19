import json
import os
from models.utils import *
import numpy as np
from sklearn import metrics as metr
import matplotlib.pyplot as plt
import matplotlib.colors as color
import seaborn as sns
import pandas as pd

TEST_JOIN = "no_attack_good_and_malware"
TEST_1 = "no_attack_goodware_apks"
TEST_2 = "no_attack_malware_apks"
TEST_3 = "feature_space_attack_2"
TEST_4 = "feature_space_attack_5"
TEST_5 = "feature_space_attack_10"

color_map = {
    "FFNN_dense_track_1": "#FF5733",
    "FFNN_ratioed_track_1": "#33FF57",
    "drebin_track_1": "#3357FF",
    "FFNN_CEL_weights_track_1": "#FF33A8",
    "FFNN_small_CEL_weights_track_1": "#A833FF",
    "secsvm_track_1": "#33FFF6",
    "FFNN_dropout_track_1": "#FFD133",
    "FFNN_small_track_1": "#FF7F33",
    "FFNN_dense_small_ratioed_track_1": "#85FF33",
    "FFNN_track_1": "#33FF85",
    "FFNN_dense_ratioed_track_1": "#33A8FF",
    "FFNN_small_dense_CEL_track_1": "#FF3333",
    "FFNN_softmax_fix_track_1": "#33FF33",
    "FFNN_only_mal_track_1": "#3333FF",
    "MyModel_track_1": "#FFAA33"
}

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

def order_models_robustness(metrics, reverse=False, test_to_order_by=TEST_2):
    aux_test_index = {TEST_2: 0, TEST_3: 1, TEST_4: 2, TEST_5: 3} 
    res = []
    aux = {}
    exceptions = ("no_attack_good_and_malware", "no_attack_goodware_apks")
    for (name, model) in metrics.items():
        if TEST_3 in model:
            aux[name] = list(test["Accuracy"] for (noun, test) in model.items() if noun not in exceptions)
    order_test_idx = aux_test_index[test_to_order_by]
    for (name, results) in sorted(aux.items(), key=lambda kv: kv[1][order_test_idx], reverse=reverse):
        if results[0] not in (0.0, 1.0):
            res += [(name, results)]
    return res


def join_all_submissions():

    test_names = {0: TEST_1, # no attack during the test in the goodware test set
                  1: TEST_2, # no attack during the test in the malware test set
                  2: TEST_3, # attacked and modified 2 features
                  3: TEST_4, # attacked and modified 5 features
                  4: TEST_5 # attacked and modified 10 features
                  }

    submissions_path = os.path.join(os.path.dirname(__file__), "submissions/")

    all_subs = {}
    filenames = os.listdir(submissions_path).sort()
    for filename in os.listdir(submissions_path):
        if filename == ".gitkeep" or "pretty" in filename:
            continue
        model_name = filename.split(".")[0][11:]
        all_subs[model_name] = {}
        with open(os.path.join(submissions_path, filename), 'r') as f:
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
    results_aux = {}

    for model in all_subs: # the first key of all_subs is a model (drevin, FFNN,...)
        results_aux[model] = {}
        for test in all_subs[model]: # each model has 5 tests, see test_names in above function for more details
            results_aux[model][test] = {
                "TP": 0,
                "FP": 0,
                "TN": 0,
                "FN": 0,
                "Accuracy": 0,
                "Score": 0
            }
            metrics = results_aux[model][test]
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
        # join both the "no_attack" tests
        goodware_test = results_aux[model][TEST_1]
        malware_test = results_aux[model][TEST_2]
        results_aux[model][TEST_JOIN] = {
            "TP": malware_test["TP"],
            "FP": goodware_test["FP"],
            "TN": goodware_test["TN"],
            "FN": malware_test["FN"],
            "Accuracy": (malware_test["TP"] + goodware_test["TN"]) / 6250,
            "Precision": malware_test["TP"] / (malware_test["TP"] + goodware_test["FP"]) if malware_test["TP"] != 0 else 0,
            "Recall":  malware_test["TP"] / (malware_test["TP"] + malware_test["FN"])  if malware_test["TP"] != 0 else 0,
            "Score": (goodware_test["Score"] * 5000 + malware_test["Score"] * 1250) / 6250
        }
        # order the tests on each model
        results[model] = {
            TEST_JOIN: results_aux[model][TEST_JOIN],
            TEST_1: results_aux[model][TEST_1],
            TEST_2: results_aux[model][TEST_2],
            TEST_3: results_aux[model][TEST_3],
            TEST_4: results_aux[model][TEST_4],
            TEST_5: results_aux[model][TEST_5]
        } if TEST_3 in results_aux[model] else {
            TEST_JOIN: results_aux[model][TEST_JOIN],
            TEST_1: results_aux[model][TEST_1],
            TEST_2: results_aux[model][TEST_2]
        }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def create_no_attack_confusion_matrices(metrics):

    conf_matrices_dir_path = os.path.join(os.path.dirname(__file__), "results/conf_matrices/")

    for model in metrics:
        test_info = metrics[model][TEST_JOIN]
        TP = test_info["TP"]
        FP = test_info["FP"]
        TN = test_info["TN"]
        FN = test_info["FN"]
        confusion_matrix = np.array([[TP, FP], [FN, TN]])
        confusion_df = pd.DataFrame(confusion_matrix, 
                        columns=["Actual Malware", "Actual Goodware"], 
                        index=["Predicted Malware", "Predicted Goodware"])
        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues", 
                    cbar=False, annot_kws={"size": 16}, linewidths=1, 
                    linecolor='black', square=True)

        plt.title(f"Confusion Matrix for {TEST_JOIN} for {model} ", fontsize=18)
        plt.xlabel("Predicted Label", fontsize=14)
        plt.ylabel("Actual Label", fontsize=14)
        
        precision = test_info["Precision"]
        recall = test_info["Recall"]
        accuracy = test_info["Accuracy"]
        plt.figtext(0.85, 0.5, f"Precision = {precision}\nRecall = {recall}\nAccuracy = {accuracy}",
                    fontsize=10, ha="left", va="center", bbox=dict(facecolor="white", alpha=0.5))
        plt.subplots_adjust(right=0.8)  # Adjust plot to make room for the text

        plt.savefig(f"{conf_matrices_dir_path}cm_{TEST_JOIN}_{model}.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_attack_confusion_matrices(metrics):
    
    conf_matrices_dir_path = os.path.join(os.path.dirname(__file__), "results/conf_matrices/")

    rows, cols = len(metrics)-1, 1
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 5 * rows), squeeze=False)

    i = 0
    for model in metrics:
        row, col = i, 0
        ax = axs[row, col]
        if TEST_3 not in metrics[model]:
            continue
        fsa_2 = metrics[model][TEST_3]
        fsa_5 = metrics[model][TEST_4]
        fsa_10 = metrics[model][TEST_5]
        confusion_matrix = [[fsa_2["TP"], fsa_5["TP"], fsa_10["TP"]],
                        [fsa_2["FN"], fsa_5["FN"], fsa_10["FN"]]]
        confusion_df = pd.DataFrame(confusion_matrix,
                                columns=["feat_space_attack_2", "feat_space_attack_5", "feat_space_attack_10"], 
                                index=["TP", "FN"])
        # Mask to only show TP row in the first heatmap (set the second row to True)
        mask_tp = np.array([[False, False, False], [True, True, True]])
        sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Greens", 
                    cbar=False, annot_kws={"size": 16}, linewidths=1, 
                    linecolor='black', square=True, ax=ax, mask=mask_tp)
    
        # Mask to only show FN row in the second heatmap (set the first row to True)
        mask_fn = np.array([[True, True, True], [False, False, False]])
        sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Reds", 
                    cbar=False, annot_kws={"size": 16}, linewidths=1, 
                    linecolor='black', square=True, ax=ax, mask=mask_fn)

        ax.set_title(f"Feature space attack results evolution for {model} ", fontsize=18)

        accuracies = (fsa_2["Accuracy"], fsa_5["Accuracy"], fsa_10["Accuracy"])
        accuracy_text = f"feat_space_attack_2 = {accuracies[0]:.2f}\nfeat_space_attack_5 = {accuracies[1]:.2f}\
                          \nfeat_space_attack_10 = {accuracies[2]:.2f}"
        ax.text(1.05, 0.5, accuracy_text, transform=ax.transAxes, fontsize=10, 
            va="center", bbox=dict(facecolor="white", alpha=0.5))
        i += 1
    
    plt.tight_layout()
    plt.savefig(f"{conf_matrices_dir_path}feature_space_attack_results_evolution_conf_matrices.png", 
                dpi=300, bbox_inches="tight")
    plt.close()


def create_attack_bar_plots(metrics):

    plots_path = os.path.join(os.path.dirname(__file__), "results/plots/")

    rows, cols = len(metrics) - 1, 1
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 5 * rows), squeeze=False)

    i = 0
    for model in metrics:
        row, col = i, 0
        ax = axs[row, col]
        if TEST_3 not in metrics[model] or metrics[model][TEST_3]["Accuracy"] == 0:
            continue
        no_attack = metrics[model][TEST_2]
        fsa_2 = metrics[model][TEST_3]
        fsa_5 = metrics[model][TEST_4]
        fsa_10 = metrics[model][TEST_5]

        x_labels = ["no_attack", "fsa_2", "fsa_5", "fsa_10"]
        values = np.array([no_attack["Accuracy"], fsa_2["Accuracy"], fsa_5["Accuracy"], fsa_10["Accuracy"]])
        ax.bar(x_labels, values)
        ax.set_ylim(0, 1)
        ax.set_title(f"Feature space attack results evolution for {model} ", fontsize=18)

        i += 1

    plt.tight_layout()
    plt.savefig(f"{plots_path}feature_space_attack_results_evolution_bar_plots.png", 
                dpi=300, bbox_inches="tight")
    plt.close()


def create_no_attack_scatter_plot(metrics : dict):

    plots_path = os.path.join(os.path.dirname(__file__), "results/plots/")

    fig, axs = plt.subplots()

    for (name, model) in metrics.items():
        axs.scatter(x=model[TEST_JOIN]["Precision"], y=model[TEST_JOIN]["Recall"],
                    label=name, c=color_map[name])

    axs.legend(bbox_to_anchor=(1.05, 1))
    axs.grid(True)
    
    axs.set_title("Precision/Recall comparison on no_attack tests")
    axs.set_xlabel("Precision")
    axs.set_ylabel("Recall")

    plt.savefig(f"{plots_path}precision-recall_comparison_no_attack_tests.png", 
                dpi=300, bbox_inches="tight")
    plt.close()


def create_attack_unique_bar_plot(metrics):

    plots_path = os.path.join(os.path.dirname(__file__), "results/plots/")

    categories = ["no_attack", "fsa_2", "fsa_5", "fsa_10"]
    bar_width = 0.5

    results = order_models_robustness(metrics, reverse=True, test_to_order_by=TEST_5)

    fig, ax = plt.subplots()
    for (name, values) in results:
        ax.bar(categories, values, bar_width, label=name, bottom=0, color=color_map[name])

    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_ylim(0, 1)

    ax.set_title("Feature_Space_Attacks accuracy comparison")
    ax.set_xlabel("Number of features attacked")
    ax.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(f"{plots_path}feature_space_attack_results_comparison_bar.png", 
                dpi=300, bbox_inches="tight")
    plt.close()


def create_no_attack_F1_measures(metrics):
    
    def F1_Measure(model_joint_results):
        precision = model_joint_results["Precision"]
        recall = model_joint_results["Recall"]
        if precision == recall == 0:
            return 0
        return round(2 * ((precision * recall) / (precision + recall)), 3)

    plots_path = os.path.join(os.path.dirname(__file__), "results/plots/")

    categories = list(model_names for model_names in metrics.keys())
    values = list(F1_Measure(model[TEST_JOIN]) for (_, model) in metrics.items())

    bars = plt.barh(categories, values)
    plt.bar_label(bars)
    plt.xlim(0, 1)

    plt.title("No_Attack F1-Measure comparison")
    plt.ylabel("Name of the model")
    plt.xlabel("F1-Measure")

    plt.tight_layout()
    plt.savefig(f"{plots_path}no_attack_f1-measure_comparison_bar.png", 
                dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    #submission_tests()
    #comparison_tests()
    all_subs = join_all_submissions()
    metrics = calculate_metrics(all_subs)

    for (name, values) in order_models_robustness(metrics, test_to_order_by=TEST_5):
        print(f"{name}: {''.join(list(' ' for _ in range(34-len(name))))} {values}")

    #create_no_attack_confusion_matrices(metrics)
    #create_no_attack_scatter_plot(metrics)
    #create_no_attack_F1_measures(metrics)

    #create_attack_confusion_matrices(metrics)
    #create_attack_bar_plots(metrics)
    #create_attack_unique_bar_plot(metrics)
