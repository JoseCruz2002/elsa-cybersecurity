import json
import os
from models.utils import *

if __name__ == "__main__":

    submissions_path = os.path.join(os.path.dirname(__file__), "submissions/")
    comparison_path = os.path.join(os.path.dirname(__file__), "results/comparison.json")
    results_path = os.path.join(os.path.dirname(__file__), "results/results.json")

    base_path = os.path.join(os.path.dirname(__file__)) 
    malware_apks_path = os.path.join(base_path, "../data/test_set_adv.zip")
    mal_apks_shas = load_sha256_list(malware_apks_path)

    results = {}
    accuracies = {}
    avg_scores = {}
    results_acc_scores = {}
    counts = {}
    for filename in os.listdir(submissions_path):
        if filename == ".gitkeep":
            continue
        accuracies[filename.split(".")[0]] = 0
        avg_scores[filename.split(".")[0]] = 0
        count = 0
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
                            if (sha256 in mal_apks_shas and test[sha256][0] == 1) or (sha256 not in mal_apks_shas and test[sha256][0] == 0) else 0
                    count += 1
                    print(f"count = {count}")
        counts[filename.split(".")[0]] = count
        results_acc_scores[filename.split(".")[0]] = [accuracies[filename.split(".")[0]] / count]
        results_acc_scores[filename.split(".")[0]] += [avg_scores[filename.split(".")[0]] / count]
    
    for model in counts:
        print(f"count for model {model}: {counts[model]}")

    with open(comparison_path, 'w') as f:
        json.dump(results, f)
    with open(results_path, "w") as f:
        json.dump(results_acc_scores, f)
    

