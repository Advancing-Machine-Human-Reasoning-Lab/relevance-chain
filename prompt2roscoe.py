"""
Filter the prompt outputs to create a roscoe input file and chain scores:
"""

import os
import re
import json
import pandas as pd
import argparse


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
    help="Prompting style. One of 'relevance-chain', 'chain-of-thought'.",
)

argparser.add_argument("--shots", type=int, default=2, help="Number of shots to use.")

argparser.add_argument("--steps", type=int, default=5, help="Number of steps to use.")

argparser.add_argument(
    "--dataset", type=str, default="test", help="Dataset to use. One of 'val', 'test'."
)

argparser.add_argument(
    "--score_type",
    type=str,
    default="chain",
    help="Type of scoring. One of 'chain', 'step'.",
)

# argparser.add_argument("--acc", type=int, default=0, help="Accuracy of the model.")


args = argparser.parse_args()
print(args)


assert args.shots in [0, 2, 4]
assert args.steps >= 0
assert args.dataset in ["val", "test"]
assert args.score_type in ["chain", "step"]


save_model_name = args.model_name
if "/" in args.model_name:
    save_model_name = args.model_name.split("/")[1]


# roscoe_input = f"roscoe_input/{args.model_name}/{args.dataset}/"
# scores = f"scores/{args.model_name}/{args.dataset}/"

if args.prompting_style == "relevance-chain":
    roscoe_input = (
        f"roscoe_input_{args.prompting_style}/{save_model_name}/{args.dataset}/"
    )
    scores = f"scores_{args.prompting_style}/{save_model_name}/{args.dataset}/"
    # load_path = f"relevance-chain/{save_model_name}/{args.dataset}/{save_model_name}_relevance-chain_shots_{args.shots}_steps_{args.steps}_{args.dataset}_{args.score_type}_{args.acc}.txt"
    load_path = f"outputs/{save_model_name}_relevance-chain_shots_{args.shots}_steps_{args.steps}_{args.dataset}_{args.score_type}.txt"

    file_name = (
        roscoe_input
        + f"ler_{save_model_name}_shots_{args.shots}_steps_{args.steps}_{args.dataset}_{args.score_type}.json"
    )
    dict_file_name = (
        scores
        + f"chain_scores_dict_{save_model_name}_shots_{args.shots}_steps_{args.steps}_{args.dataset}_{args.score_type}.txt"
    )

elif args.prompting_style == "chain-of-thought":
    roscoe_input = (
        f"roscoe_input_{args.prompting_style}/{save_model_name}/{args.dataset}/"
    )
    scores = f"scores_{args.prompting_style}/{save_model_name}/{args.dataset}/"
    # load_path = f"chain-of-thought/{save_model_name}/{args.dataset}/{save_model_name}_chain-of-thought_shots_{args.shots}_{args.dataset}_{args.acc}.txt"
    load_path = f"outputs/{save_model_name}_chain-of-thought_shots_{args.shots}_{args.dataset}.txt"

    file_name = (
        roscoe_input
        + f"ler_{save_model_name}_shots_{args.shots}_{args.dataset}.json"
    )

if not os.path.exists(roscoe_input):
    os.makedirs(roscoe_input)
if not os.path.exists(scores):
    os.makedirs(scores)


def filter_output(output_list):
    filtered_list = ""
    first_n = False
    steps = 0
    for l in output_list:
        if l.startswith("Step 1") and first_n == True:
            break
        if l.startswith("Step 1"):
            first_n = True
        if l.startswith("Step "):
            l = l.replace("\n", " ")
            filtered_list = filtered_list + l.split(": ")[1]
            steps = steps + 1
        else:
            continue

    if filtered_list == "":
        filtered_list = output_list[0]
        filtered_list = filtered_list.replace("\n", "")

    step_count = steps
    to_return = filtered_list.strip()

    pp = re.compile(r"(highly relevant|partially relevant|partial relevant|irrelevant)")
    match = pp.findall(output_list[-1].lower())

    if match != []:
        prediction = match[0]
        if prediction.startswith("partial relevant"):
            prediction = "partially relevant"
    elif match == []:
        prediction = "irrelevant"

    return to_return, prediction, step_count


def filter_hug_outputs(output_list):
    pattern = r"Step \d+: "
    to_return = re.sub(pattern, "", output_list[0])
    to_return = to_return.replace("\n", "")

    # pattern = r'Step (\d+): '
    # matches = re.findall(pattern, output_list[0])
    # to_return = re.sub(pattern, '', output_list[0])
    # step_count = len(matches)

    score = output_list[-2]

    pp = re.compile(r"(highly relevant|partially relevant|partial relevant|irrelevant)")
    match = pp.findall(output_list[-1].lower())

    if match != []:
        prediction = match[0]
        if prediction.startswith("partial relevant"):
            prediction = "partially relevant"
    elif match == []:
        prediction = "irrelevant"

    return to_return, prediction, score


def filter_step_score(score_list):
    scores = []
    for step_score in score_list:
        if step_score.startswith("Step "):
            filtered_l = step_score.split(": ")[-1]
            # digits = re.findall(r'\d+', filtered_l)
            digits = re.findall(r"[-+]?(?:\d*\.*\d+)", filtered_l)

            if digits != []:
                score = float(digits[-1])
                if 1 <= score <= 10:
                    scores.append(score)

        else:
            filtered_l = step_score.replace("\n", " ")
            # digits = re.findall(r'\d+', filtered_l)
            digits = re.findall(r"[-+]?(?:\d*\.*\d+)", filtered_l)

            if digits != []:
                score = float(digits[-1])

                if 1 <= score <= 10:
                    scores.append(score)

    return scores


def filter_cot_output(output_list):
    filtered_list = ""

    for l in output_list[:-1]:
        l = l.replace("\n", "")
        filtered_list = filtered_list + l + " "

    # Remove " " at end of the string:
    to_return = filtered_list.strip()

    pp = re.compile(r"(highly relevant|partially relevant|partial relevant|irrelevant)")
    match = pp.findall(output_list[-1].lower())

    if match != []:
        prediction = match[0]
        if prediction.startswith("partial relevant"):
            prediction = "partially relevant"
        # if prediction.startswith("fully relevant") or prediction.startswith("High relevant") or prediction.startswith("very relevant"):
        # prediction="highly relevant"
        # if prediction.startswith("impartiant") or prediction.startswith("not possible"):
        # prediction="irrelevant"

    elif match == []:
        print("Warning. There is no match!")
        exit()

    return to_return, prediction


if args.dataset == "val":
    data = pd.read_csv("dataset/val_data.tsv", sep="\t", index_col=None)
elif args.dataset == "test":
    data = pd.read_csv("dataset/test_data.tsv", sep="\t", index_col=None)


dict = {}
if args.prompting_style == "relevance-chain":
    chain_scores = []
    no_step_filter = []

with open(load_path, "r") as f:
    with open(file_name, "w") as outfile:
        steps_list = []
        for line in f:
            p = re.compile(r"### \d+ ###")
            match = p.findall(line)

            if match != []:
                p2 = re.compile(r"\d+")
                match2 = p2.findall(match[0])
                index = int(match2[0])

                if steps_list != []:
                    steps_list = [item for item in steps_list if item != "\n"]

                    if args.prompting_style == "relevance-chain":
                        if save_model_name in ["flan-t5-xxl", "flan-alpaca-xxl"]:
                            reasoning_steps, pred, chain_score = filter_hug_outputs(
                                steps_list
                            )
                            try:
                                chain_score = float(chain_score.replace("\n", ""))
                            except:
                                chain_score = 0
                            chain_scores.append(chain_score)

                        elif save_model_name in ["gpt-3.5-turbo", "gpt-4"]:
                            reasoning_steps, pred, step_count = filter_output(
                                steps_list
                            )

                            if step_count == 0:
                                no_step_filter.append(index - 1)
                                chain_scores.append(0)

                            else:
                                if args.score_type == "step":
                                    chain_score = filter_step_score(
                                        steps_list[-1 - step_count : -1]
                                    )  # eskisi -3
                                elif args.score_type == "chain":
                                    chain_score = steps_list[-2]  # eskisi -4
                                    chain_score = float(chain_score.replace("\n", ""))

                                chain_scores.append(chain_score)

                    elif args.prompting_style == "chain-of-thought":
                        reasoning_steps, pred = filter_cot_output(steps_list)

                    gpt = f"{reasoning_steps} The answer is {pred}."
                    fact = data.iloc[index - 1]["fact"]
                    evidence = data.iloc[index - 1]["evidence"]
                    score = data.iloc[index - 1]["score"]

                    if score == 0:
                        hypothesis = "irrelevant"
                    elif score == 1:
                        hypothesis = "partially relevant"
                    elif score == 2:
                        hypothesis = "highly relevant"
                    else:
                        print("There is no score with 0,1 or 2!")
                        exit()

                    premise = f"Fact: {fact} Report: {evidence} Answer how relevant the Report is as evidence of the Fact."

                    dict["premise"] = premise
                    dict["hypothesis"] = hypothesis
                    dict["gpt-3"] = gpt
                    dict["answer"] = "yes"

                    # dict["explanation_1"] = "."
                    # dict["explanation_2"] = "."
                    # dict["explanation_3"] = "."

                    print(dict)

                    json.dump(dict, outfile)
                    outfile.write("\n")

                dict["key"] = match2[0]
                print(index)
                steps_list = []

            else:
                steps_list.append(line)

        steps_list = [item for item in steps_list if item != "\n"]

        if args.prompting_style == "relevance-chain":
            if save_model_name in ["flan-t5-xxl", "flan-alpaca-xxl"]:
                reasoning_steps, pred, chain_score = filter_hug_outputs(steps_list)

                try:
                    chain_score = float(chain_score.replace("\n", ""))
                except:
                    chain_score = 0

                chain_scores.append(chain_score)

            elif save_model_name in ["gpt-3.5-turbo", "gpt-4"]:
                reasoning_steps, pred, step_count = filter_output(steps_list)

                if step_count == 0:
                    no_step_filter.append(index)
                    chain_scores.append(0)
                else:
                    chain_score = steps_list[-2]
                    if args.score_type == "step":
                        chain_score = filter_step_score(
                            steps_list[-1 - step_count : -1]
                        )
                    elif args.score_type == "chain":
                        chain_score = float(chain_score.replace("\n", ""))
                    chain_scores.append(chain_score)

        elif args.prompting_style == "chain-of-thought":
            reasoning_steps, pred = filter_cot_output(steps_list)

        gpt = f"{reasoning_steps} The answer is {pred}."

        fact = data.iloc[index]["fact"]
        evidence = data.iloc[index]["evidence"]
        score = data.iloc[index]["score"]

        if score == 0:
            hypothesis = "irrelevant"
        elif score == 1:
            hypothesis = "partially relevant"
        elif score == 2:
            hypothesis = "highly relevant"
        else:
            print("There is no score with 0,1 or 2!")
            exit()

        premise = f"Fact: {fact} Report: {evidence} Answer how relevant the Report is as evidence of the Fact."

        dict["key"] = match2[0]
        dict["premise"] = premise
        dict["hypothesis"] = hypothesis
        dict["gpt-3"] = gpt
        dict["answer"] = "yes"

        # dict["explanation_1"] = "."
        # dict["explanation_2"] = "."
        # dict["explanation_3"] = "."

        print(dict)
        json.dump(dict, outfile)


if args.prompting_style == "relevance-chain":
    chain_scores_dict = {}
    for i in range(len(chain_scores)):
        chain_scores_dict[i] = chain_scores[i]

    f = open(dict_file_name, "w")
    f.write(str(chain_scores_dict))
    f.close()

    print("Samples without chain: ", no_step_filter)
