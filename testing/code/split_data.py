import json
import os
import pdb
import random
from pathlib import Path
from typing import List

import fire
import load_questions
import numpy as np
import prompt_models
import pyminizip
import torch
from guidance import models
from tqdm import tqdm

SPLITS = [0.75, 0.1, 0.15]
QSPLITS = [0.8, 0.2]
NAMES = ["train", "dev"]
TEST_NAMES = [f"test{i}" for i in range(20)]
NAMES = NAMES + TEST_NAMES

TRAIN_SPLIT = set({"Problem-train_Obfus-train_Question-train"})
DEV_SPLIT = set({"Problem-dev_Obfus-train_Question-train"})
PW = "lingoly"
SEED = 104


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Pytorch
    torch.mps.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_data(data, splits):
    size = len(data)
    curr = 0
    splitted = []
    for split in splits:
        split_size = int(size * split)
        splitted.append(data[curr : curr + split_size])
        curr += split_size

    if curr < size:
        splitted[-1].extend(data[curr:])

    return splitted


def get_data_splits(data, splits_list, idx=1, key="question_n"):
    res = []
    for split in splits_list:
        tmp = []
        for d in data:
            if d[idx][key] in split:
                # print(f"Found: {d[idx][key]} in {split}")
                tmp.append(d)
        res.append(tmp)
    return res


def get_question_versions(data, q):
    filtered = []
    for d in data:
        if d["overall_question_n"] == q:
            if d["obfuscated"] == "False":
                # place original problem at beginning
                filtered = [d] + filtered
            else:
                filtered.append(d)

    return filtered


def pipeline(
    splits: List[int] = SPLITS,
    data_zip: str = "../data/benchmark_obf.zip",
    data_filename: str = "../data/test_obf.jsonl",
    exclude: List[int] = None,
    pw: str = PW,
    cot: bool = False,
    no_context: bool = False,
    outfolder: str = "../data/splits",
    outfile: str = "modelname_questionsname.json",
    seed: int = SEED,
):
    set_seeds(seed)

    # Loading data
    print("Loading questions")
    pyminizip.uncompress(data_zip, PW, "../data/", 0)
    os.chdir("../code")
    question_sheets = []
    q_sets = set()
    with open(data_filename) as f:
        for line in f:
            q = json.loads(line)
            q_sets.add(q["overall_question_n"])
            question_sheets.append(q)
    Path.unlink(data_filename)
    print("All problems in this dataset:")
    print(q_sets)
    print()

    # Split
    tot = sum(splits)
    if tot > 1.0:
        raise ValueError("total splits cannot exceed 1.0")

    num_questions = len(q_sets)

    pool = q_sets
    if exclude is not None:
        # TODO: re-add to test split
        pool.difference_update(set(exclude))

    q_splits = split_data(list(pool), splits)

    print("Test problems:")
    print(q_splits[2])

    model = "Ours"
    folder = Path(outfolder)
    folder.mkdir(exist_ok=True)

    final_data = {}
    for q, ques_split in enumerate(q_splits):
        for ques in ques_split:
            print(f"processing question: {ques}")
            ques_versions = get_question_versions(question_sheets, ques)
            if len(ques_versions) == 0:
                print(f"Warning: question {ques} was not found in data")
                continue
            # Shuffle obfuscations of a problem
            random.shuffle(ques_versions)
            splitted = split_data(ques_versions, splits)

            if len(splitted[0]) == 0:
                print(f"Warning: no data in split 0. Skipping")
                continue
            # print("Generating new question splits")
            splitted_sub_set = []
            one_example_idx, one_example_meta = load_questions.load_all_questions_only(
                [splitted[0][0]]
            )
            for subq in one_example_meta:
                splitted_sub_set.append(subq["question_n"])
            splitted_sub_set = split_data(splitted_sub_set, QSPLITS)

            for o, split in enumerate(splitted):
                q_idx, meta = load_questions.load_all_questions_only(split)

                # Get list of all questions
                data = [(a, b) for a, b, in zip(q_idx, meta)]
                splitted_sub = get_data_splits(data, splitted_sub_set)
                # pdb.set_trace()
                for sq, sub_q in enumerate(splitted_sub):
                    key = f"Problem-{NAMES[q]}_Obfus-{NAMES[o]}_Question-{NAMES[sq]}"
                    final_data[key] = final_data.get(key, [])
                    final_data[key].extend(sub_q)

    # Write files separately
    all_data = []
    for k in final_data.keys():
        if len(final_data[k]) == 0:
            print(f"Warning: empty split: {k}")
            continue
        tmp_file = f"{folder}/{k}.jsonl"
        # save prompts temp
        with open(tmp_file, "w") as f:
            for example in final_data[k]:
                i, r = example
                line = {"index": i, "split_key": k, "question_details": r}
                all_data.append(line)
                f.write(json.dumps(line) + "\n")
    print("Data splitting done")

    # Write all data in out file
    print(f"Saving all {len(all_data)} prompts")
    all_data_file = f"{folder}/benchmark.jsonl"
    with open(all_data_file, "w") as f:
        for example in all_data:
            f.write(json.dumps(example) + "\n")
    print(f"All data is saved in {all_data_file}")

    # Write final merged splits
    dev_set = {}
    test_set = {}
    train_file = f"{folder}/train.jsonl"
    with open(train_file, "w") as f:
        for k in final_data.keys():
            if k in TRAIN_SPLIT:
                for example in final_data[k]:
                    i, r = example
                    line = {"index": i, "split_key": k, "question_details": r}
                    # prompt, ans, i, r = example
                    # line = {"prompt": prompt, "completion": ans, "index": i, "split_key": k, "rest": r}
                    # prompt, ans, i = example
                    # line = {"prompt": prompt, "completion": ans, "index": i, "split_key": k}
                    f.write(json.dumps(line) + "\n")
            else:
                dev_set[k] = final_data[k]

    dev_file = f"{folder}/dev.jsonl"
    with open(dev_file, "w") as f:
        for k in dev_set.keys():
            if k in DEV_SPLIT:
                for example in dev_set[k]:
                    i, r = example
                    line = {"index": i, "split_key": k, "question_details": r}
                    # prompt, ans, i, r = example
                    # line = {"prompt": prompt, "completion": ans, "index": i, "split_key": k, "rest": r}
                    # prompt, ans, i = example
                    # line = {"prompt": prompt, "completion": ans, "index": i, "split_key": k}
                    f.write(json.dumps(line) + "\n")
            else:
                test_set[k] = dev_set[k]

    test_file = f"{folder}/test.jsonl"
    with open(test_file, "w") as f:
        for k in test_set.keys():
            for example in test_set[k]:
                i, r = example
                line = {"index": i, "split_key": k, "question_details": r}
                # prompt, ans, i, r = example
                # line = {"prompt": prompt, "completion": ans, "index": i, "split_key": k, "rest": r}
                # prompt, ans, i = example
                # line = {"prompt": prompt, "completion": ans, "index": i, "split_key": k}
                f.write(json.dumps(line) + "\n")

    # Compress files
    pyminizip.compress(all_data_file, None, all_data_file + ".zip", pw, 0)
    pyminizip.compress(train_file, None, train_file + ".zip", pw, 0)
    pyminizip.compress(dev_file, None, dev_file + ".zip", pw, 0)
    pyminizip.compress(test_file, None, test_file + ".zip", pw, 0)

    Path.unlink(all_data_file)
    Path.unlink(train_file)
    Path.unlink(dev_file)
    Path.unlink(test_file)
    print("Finished!")


if __name__ == "__main__":
    fire.Fire(pipeline)
