"""
Calculates correlation between roscoe scores, chain scoreas and labels:
"""

import numpy as np
import pandas as pd
import argparse
import ast

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import hmean


def step_score(step_scores):
    if type(step_scores) == list:
        if args.step_score == "mean":
            chain_score = np.mean(step_scores)
        elif args.step_score == "min":
            chain_score = np.min(step_scores)
        elif args.step_score == "hmean":
            chain_score = hmean(step_scores)
    else:
        chain_score = step_scores
    return chain_score


argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--model_name",
    type=str,
    default="gpt-3.5-turbo",
    help="Model name. Can be any one of: 'gpt-3.5', gpt-3.5-turbo', 'gpt-4', or any huggingface models.",
)

argparser.add_argument(
    "--prompting_style",
    type=str,
    default="relevance-chain",
    help="Prompting style. One of 'relevance-chain', 'vanilla', 'chain-of-thought'.",
)

argparser.add_argument("--shots", type=int, default=2, help="Number of shots to use.")

argparser.add_argument("--steps", type=int, default=5, help="Number of steps to use.")

argparser.add_argument(
    "--score_type",
    type=str,
    default="chain",
    help="Type of scoring. One of 'chain', 'step'.",
)


argparser.add_argument(
    "--use_pred",
    action=argparse.BooleanOptionalAction,
    help="Trains with model predictions if added. ",
)

argparser.add_argument(
    "--no_chain",
    action=argparse.BooleanOptionalAction,
    help="Trains without chain score if added. ",
)

argparser.add_argument(
    "--step_score",
    type=str,
    default="mean",
    help="Type of scoring strategy for steps. Works only if score_type is 'step'. One of 'mean', 'min'.",
)

argparser.add_argument(
    "--extern_scorer",
    type=str,
    default=None,
    help="Model name. Can be any one of: 'gpt-3.5-turbo', 'gpt-4', or any huggingface models.",
)

argparser.add_argument(
    "--extern_score_type",
    type=str,
    default="chain",
    help="Type of scoring. One of 'chain', 'step'.",
)

args = argparser.parse_args()
print("Arguments:\n", args)

assert args.shots in [0, 2, 4]
assert args.steps >= 0
assert args.score_type in ["chain", "step"]
assert args.step_score in ["mean", "min", "hmean"]
assert args.extern_score_type in ["chain", "step"]


if args.extern_scorer != None:
    extern_scorer = args.extern_scorer
    if "/" in args.extern_scorer:
        extern_scorer = args.extern_scorer.split("/")[1]

    extern_scorer_val_file = f"extern_scorer/{extern_scorer}/{extern_scorer}_{args.model_name}_shots_{args.shots}_steps_{args.steps}_val_{args.score_type}_{args.extern_score_type}.tsv"


    extern_scorer_val_score = pd.read_csv(
        extern_scorer_val_file, sep="\t", index_col=None
    )[f"{args.extern_score_type}_score"]

    extern_scorer_test_file = f"extern_scorer/{extern_scorer}/{extern_scorer}_{args.model_name}_shots_{args.shots}_steps_{args.steps}_test_{args.score_type}_{args.extern_score_type}.tsv"

    extern_scorer_test_score = pd.read_csv(
        extern_scorer_test_file, sep="\t", index_col=None
    )[f"{args.extern_score_type}_score"]


val_dataset = "val"

if args.prompting_style == "relevance-chain":
    roscoe_file = f"scores_{args.prompting_style}/{args.model_name}/{val_dataset}/scores_ler_{args.model_name}_shots_{args.shots}_steps_{args.steps}_{val_dataset}_{args.score_type}.tsv"
    load_labels = f"results/{args.model_name}_relevance-chain_shots_{args.shots}_steps_{args.steps}_{val_dataset}_{args.score_type}.tsv"
    chain_score_dict = f"scores_{args.prompting_style}/{args.model_name}/{val_dataset}/chain_scores_dict_{args.model_name}_shots_{args.shots}_steps_{args.steps}_{val_dataset}_{args.score_type}.txt"


elif args.prompting_style == "chain-of-thought":
    roscoe_file = f"scores_{args.prompting_style}/{args.model_name}/{val_dataset}/scores_ler_{args.model_name}_shots_{args.shots}_{val_dataset}.tsv"
    load_labels = f"results/{args.model_name}_chain-of-thought_shots_{args.shots}_{val_dataset}.tsv"

roscoe_scores = pd.read_csv(
    roscoe_file, delimiter=r"\s+", engine="python", index_col="ID"
)

labels = pd.read_csv(
    load_labels, delimiter=r"\t", engine="python", index_col="index"
).label

predictions = pd.read_csv(
    load_labels, delimiter=r"\t", engine="python", index_col="index"
).prediction
le = preprocessing.LabelEncoder()

le.fit(["irrelevant", "partially relevant", "highly relevant"])

labels = le.transform(labels)
classes = dict(zip(le.classes_, le.transform(le.classes_)))


if args.prompting_style == "relevance-chain":
    if args.extern_scorer == None:
        with open(chain_score_dict) as f:
            data = f.read()
        score_dict = ast.literal_eval(data)
        chain_scores = list(score_dict.values())
    else:
        if args.extern_score_type == "chain":
            extern_scorer_val_score = extern_scorer_val_score.astype(float)
            chain_scores = list(extern_scorer_val_score)

        elif args.extern_score_type == "step":
            chain_scores = [ast.literal_eval(sc) for sc in extern_scorer_val_score]
            for i in range(len(chain_scores)):
                chain_scores[i] = step_score(chain_scores[i])


x_train, y_train = roscoe_scores, labels
if args.prompting_style == "relevance-chain" and not args.no_chain:
    x_train["chain_score"] = chain_scores

if args.use_pred:
    x_train["prediction"] = le.transform(predictions)



###
# CLASSIFIER:
###

min_max_scaling = True
feature_selection = True
number_of_features = 10

if min_max_scaling:
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train[x_train.columns] = min_max_scaler.fit_transform(x_train[x_train.columns])

if feature_selection:
    feat_sel = SelectKBest(f_classif, k=number_of_features)
    x_train_transformed = feat_sel.fit_transform(x_train, y_train)
    # Get selected columns:
    cols_idxs = feat_sel.get_support(indices=True)
    features_df_new = x_train.iloc[:, cols_idxs].columns.values
    x_train = x_train_transformed


# clf = SVC(kernel="poly", C=1, degree=3)
clf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42)


clf.fit(x_train, y_train)


######
# TEST
######

test_dataset = "test"

if args.prompting_style == "relevance-chain":
    roscoe_file = f"scores_{args.prompting_style}/{args.model_name}/{test_dataset}/scores_ler_{args.model_name}_shots_{args.shots}_steps_{args.steps}_{test_dataset}_{args.score_type}.tsv"
    load_labels = f"results/{args.model_name}_relevance-chain_shots_{args.shots}_steps_{args.steps}_{test_dataset}_{args.score_type}.tsv"
    chain_score_dict = f"scores_{args.prompting_style}/{args.model_name}/{test_dataset}/chain_scores_dict_{args.model_name}_shots_{args.shots}_steps_{args.steps}_{test_dataset}_{args.score_type}.txt"


elif args.prompting_style == "chain-of-thought":
    roscoe_file = f"scores_{args.prompting_style}/{args.model_name}/{test_dataset}/scores_ler_{args.model_name}_shots_{args.shots}_{test_dataset}.tsv"
    load_labels = f"results/{args.model_name}_chain-of-thought_shots_{args.shots}_{test_dataset}.tsv"


roscoe_scores = pd.read_csv(
    roscoe_file, delimiter=r"\s+", engine="python", index_col="ID"
)
labels = pd.read_csv(
    load_labels, delimiter=r"\t", engine="python", index_col="index"
).label


predictions = pd.read_csv(
    load_labels, delimiter=r"\t", engine="python", index_col="index"
).prediction

labels = le.transform(labels)

if args.prompting_style == "relevance-chain":
    if args.extern_scorer == None:
        with open(chain_score_dict) as f:
            data = f.read()
        score_dict = ast.literal_eval(data)
        chain_scores = list(score_dict.values())
    else:
        if args.extern_score_type == "chain":
            extern_scorer_test_score = extern_scorer_test_score.astype(float)
            chain_scores = list(extern_scorer_test_score)

        elif args.extern_score_type == "step":
            chain_scores = [ast.literal_eval(sc) for sc in extern_scorer_test_score]
            for i in range(len(chain_scores)):
                chain_scores[i] = step_score(chain_scores[i])


x_test, y_test = roscoe_scores, labels
if args.prompting_style == "relevance-chain" and not args.no_chain:
    x_test["chain_score"] = chain_scores
if args.use_pred:
    x_test["prediction"] = le.transform(predictions)


if min_max_scaling:
    x_test[x_test.columns] = min_max_scaler.transform(x_test[x_test.columns])

if feature_selection:
    x_test = feat_sel.transform(x_test)

y_pred = clf.predict(x_test)

conf_mat = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        classes["irrelevant"],
        classes["partially relevant"],
        classes["highly relevant"],
    ],
)
print("Confusion matrix: rows:true | columns:predicted\n", conf_mat)

acc = accuracy_score(y_test, y_pred)
print("Test Accuracy: ", acc)
