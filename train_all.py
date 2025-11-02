# train_all_models.py
import subprocess
scripts = [
    "train_linear_regression.py",
    "train_naive_bayes.py",
    "train_knn.py",
    "train_svm.py",
    "train_decision_tree.py",
    "train_ann.py"
]

for s in scripts:
    print("Running", s)
    subprocess.run(["python", s], check=True)
    print("Finished", s)
