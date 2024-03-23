"""
Get prompt outputs.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def chain_check(output):
    output_list = output.splitlines()

    for l in output_list:
        if l.startswith("Step 1:"):
            return True
    return False


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
    "--dataset", type=str, default="test", help="Dataset to use. One of 'val', 'test'."
)

argparser.add_argument(
    "--score_type",
    type=str,
    default="chain",
    help="Type of scoring. One of 'chain', 'step'.",
)

args = argparser.parse_args()
print(args)


assert args.prompting_style in ["relevance-chain", "chain-of-thought", "vanilla"]
assert args.shots in [2]
assert args.steps >= 0
assert args.dataset in ["val", "test"]
assert args.score_type in ["chain", "step"]


match args.model_name:
    case "gpt-3.5-turbo" | "gpt-3" | "gpt-4":
        import openai
        from dotenv import load_dotenv

        load_dotenv("./.env")
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    case _:
        from transformers import AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        text_generation_pipeline = pipeline(model=args.model_name, device_map="auto")


def get_output(msg):
    match args.model_name:
        case args.model_name if args.model_name in ["gpt-3.5-turbo", "gpt-4"]:
            while True:
                try:
                    output = openai.ChatCompletion.create(  # get model output
                        model=args.model_name,  # use turbo model
                        messages=msg,  # use prompt
                        temperature=1.0,
                        top_p=1,
                        max_tokens=1024,
                    )["choices"][0]["message"]["content"]

                    return output
                except openai.error.InvalidRequestError:
                    raise

        case _:
            while True:
                return text_generation_pipeline(
                    msg,
                    max_length=1024,
                    temperature=1,
                    top_p=1,
                    do_sample=True,
                    num_return_sequences=1,
                )[0]["generated_text"]


results_folder = "results/"
outputs_folder = "outputs/"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)


save_model_name = args.model_name
if "/" in args.model_name:
    save_model_name = args.model_name.split("/")[1]

if args.prompting_style == "relevance-chain":
    file_name = f"{save_model_name}_{args.prompting_style}_shots_{args.shots}_steps_{args.steps}_{args.dataset}_{args.score_type}.tsv"
else:
    file_name = f"{save_model_name}_{args.prompting_style}_shots_{args.shots}_{args.dataset}.tsv"

save_path = results_folder + file_name


results_df = pd.DataFrame(columns=["index", "prediction", "label"])
if os.path.exists(save_path):
    results_df = pd.read_csv(save_path, sep="\t", index_col=None)

if args.dataset == "val":
    data = pd.read_csv("dataset/val_data.tsv", sep="\t", index_col=None)
elif args.dataset == "test":
    data = pd.read_csv("dataset/test_data.tsv", sep="\t", index_col=None)


pbar = tqdm(total=len(data))
pbar.update(len(results_df))


for i in range(len(results_df), len(data)):
    fact = data.iloc[i]["fact"]
    evidence = data.iloc[i]["evidence"]
    score = data.iloc[i]["score"]

    shots = ""
    if args.shots in [1, 2, 4]:
        f = open(f"shots/{args.shots}-shots-{args.prompting_style}.txt", "r")
        shots = f.read()

    match args.model_name:
        case args.model_name if args.model_name in ["gpt-3.5-turbo", "gpt-4"]:
            match args.prompting_style:
                case "relevance-chain":
                    msg = [
                        {
                            "role": "system",
                            "content": "You are a legal reasoning system. Given the Report and the Fact, you must find the strongest relevance chain between the Report and the Fact. Then answer how relevant the Report is as evidence of the Fact.",
                        }
                    ]

                    msg.append(
                        {
                            "role": "user",
                            "content": f"{shots}Fact: {fact}\n\nReport: {evidence}\n\nAnswer how relevant the Report is as evidence of the Fact. Thus, find the strongest relevance chain between the Report and the Fact in {args.steps} steps or less. Use 'makes it more likely' or 'makes it less likely' relation in each step.",
                        }
                    )

                    answer = get_output(msg)
                    has_step = chain_check(answer)

                    msg.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                    if has_step == False:
                        msg.append(
                            {
                                "role": "user",
                                "content": f"There is a relevance chain between the Report and the Fact even if the Report is irrelevant. Find a weak relevance chain. But don't change your decision that the Report is irrelevant.",
                            }
                        )

                        answer = get_output(msg)
                        msg.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                    if args.score_type == "chain":
                        msg.append(
                            {
                                "role": "user",
                                "content": f"For the latest scenario, give a score for the strength of the relevance chain. Use a scale of 1 to 10, where 1 represents the weakest chain, and 10 represents the strongest. Write nothing but the score.",
                            }
                        )
                    elif args.score_type == "step":
                        msg.append(
                            {
                                "role": "user",
                                "content": f"For the latest scenario, give a score for the strength of each step in the relevance chain. Use a scale of 1 to 10, where 1 represents the weakest chain, and 10 represents the strongest. Write nothing but the score.",
                            }
                        )

                    answer2 = get_output(msg)

                    msg.append(
                        {
                            "role": "assistant",
                            "content": answer2,
                        }
                    )

                    msg.append(
                        {
                            "role": "user",
                            "content": f"So, answer how relevant the Report is as evidence of the Fact. Just answer as 'Highly relevant', 'Partially relevant', or 'Irrelevant'. Write nothing else.",  #
                        }
                    )

                case "chain-of-thought":
                    msg = [
                        {
                            "role": "system",
                            "content": "You are a legal reasoning system. Given the Report and the Fact, you must answer how relevant the Report is as evidence of the Fact.",
                        }
                    ]

                    msg.append(
                        {
                            "role": "user",
                            "content": f"{shots}Fact: {fact}\n\nReport: {evidence}\n\nAnswer how relevant the Report is as evidence of the Fact. Let's think step by step.",  ##best
                        }
                    )

                    answer = get_output(msg)

                    msg.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )
                    msg.append(
                        {
                            "role": "user",
                            "content": "So, answer how relevant the Report is as evidence of the Fact. Just answer as 'Highly relevant', 'Partially relevant', or 'Irrelevant'. Write nothing else.",
                        }
                    )

                case "vanilla":
                    msg = [
                        {
                            "role": "system",
                            "content": "You are a legal reasoning system. Given the Report and the Fact, you must answer how relevant the Report is as evidence of the Fact.",
                        }
                    ]

                    msg.append(
                        {
                            "role": "user",
                            "content": f"{shots}Fact: {fact}\n\nReport: {evidence}\n\nAnswer how relevant the Report is as evidence of the Fact. Just answer as 'Highly relevant', 'Partially relevant', or 'Irrelevant'. Write nothing else.",  ##best
                        }
                    )

        case _:
            match args.prompting_style:
                case "relevance-chain":
                    system = "You are a legal reasoning system. Given the Report and the Fact, you must find the strongest relevance chain between the Report and the Fact. Then answer how relevant the Report is as evidence of the Fact."
                    msg = f"{system}\n\n{shots}Fact: {fact}\n\nReport: {evidence}\n\nAnswer how relevant the Report is as evidence of the Fact. Thus, find the strongest relevance chain between the Report and the Fact in {args.steps} steps or less. Use 'makes it more likely' or 'makes it less likely' relation in each step."

                    answer = get_output(msg)

                    msg = f"{msg}\n\n{answer}\nFor the latest scenario, give a score for the strength of the relevance chain. Use a scale of 1 to 10, where 1 represents the weakest chain, and 10 represents the strongest. Write nothing but the score"

                    answer2 = get_output(msg)

                    msg = f"{msg}\n\n{answer2}\nSo, answer how relevant the Report is as evidence of the Fact. Just answer as 'Highly relevant', 'Partially relevant', or 'Irrelevant'. Write nothing else."

                case "chain-of-thought":
                    system = "You are a legal reasoning system. Given the Report and the Fact, you must answer how relevant the Report is as evidence of the Fact."
                    msg = f"{system}\n\n{shots}Fact: {fact}\n\nReport: {evidence}\n\nAnswer how relevant the Report is as evidence of the Fact. Let's think step by step."
                    answer = get_output(msg)
                    msg = f"{msg}\n\n{answer}\nSo, answer how relevant the Report is as evidence of the Fact. Just answer as 'Highly relevant', 'Partially relevant', or 'Irrelevant'. Write nothing else."

    output = get_output(msg)

    if args.prompting_style in ["relevance-chain", "chain-of-thought"]:
        if args.prompting_style == "relevance-chain":
            output_path = f"outputs/{save_model_name}_{args.prompting_style}_shots_{args.shots}_steps_{args.steps}_{args.dataset}_{args.score_type}.txt"
        else:
            output_path = f"outputs/{save_model_name}_{args.prompting_style}_shots_{args.shots}_{args.dataset}.txt"
        with open(output_path, "a") as f:
            if args.prompting_style == "relevance-chain":
                if args.model_name in ["gpt-3.5-turbo", "gpt-4"]:
                    if len(msg) == 6:
                        output_txt = f"### {i} ###\n\n{msg[2]['content']}\n\n{msg[4]['content']}\n\n{output}\n\n"
                    elif len(msg) == 8:
                        output_txt = f"### {i} ###\n\n{msg[2]['content']}\n\n{msg[4]['content']}\n\n{msg[6]['content']}\n\n{output}\n\n"
                else:
                    output_txt = f"### {i} ###\n\n{answer}\n\n{answer2}\n\n{output}\n\n"

            elif args.prompting_style == "chain-of-thought":
                if args.model_name in ["gpt-3.5-turbo", "gpt-4"]:
                    output_txt = f"### {i} ###\n\n{msg[2]['content']}\n\n{output}\n\n"
                else:
                    output_txt = f"### {i} ###\n\n{answer}\n\n{output}\n\n"

            f.write(output_txt)

    pp = re.compile(r"(highly relevant|partially relevant|irrelevant)")
    match = pp.findall(output.lower())

    if match != []:
        output = match[0]

    if score == 0:
        label = "irrelevant"
    elif score == 1:
        label = "partially relevant"
    elif score == 2:
        label = "highly relevant"
    else:
        print("Error. Label not in list!")

    row_result = pd.DataFrame(
        {"index": i, "prediction": output, "label": label}, index=[0]
    )

    results_df = pd.concat([results_df, row_result], axis=0, ignore_index=True)
    results_df.to_csv(save_path, sep="\t", index=False)

    pbar.update(1)

    count = results_df.apply(lambda row: row["prediction"] == row["label"], axis=1)


count = results_df.apply(lambda row: row["prediction"] == row["label"], axis=1)
accuracy = sum(count) * 100 / len(results_df)
print(results_df)
print(accuracy)
