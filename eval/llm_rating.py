
import os
import re
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from api.chat_completion_api import call_api

QA_RATING_PROMPT = """You are asked to evaluate the quality of the AI assistant's answers to user question as an impartial judge, \
and your evaluation should take into account factors including correctness (high priority), and comprehensiveness (whether the \
assistant's answer covers all points). Read the AI assistant's answer and compare against the reference answer, and give an overall \
integer rating in 1, 2, 3 (1 = wrong or irrelevant, 2 = partially correct, 3 = correct and comprehensive) based on the above principles, \
strictly in the following format:"[[rating]]", e.g. "[[2]]". 

Question: {question}
Reference answer: {reference}
Assistant's answer: {prediction}
Rating: """

SUMMARIZATION_RATING_PROMPT = """You are asked to evaluate the quality of the AI assistant's generated summary as an impartial judge, \
and your evaluation should take into account factors including correctness (high priority), comprehensiveness (whether the assistant's \
summary covers all points), and coherence. Read the AI assistant's summary and compare against the reference summary, and give an overall \
integer rating in on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the evaluation criteria, strictly in the \
following format:"[[rating]]", e.g. "[[3]]". 

Reference summary: 
{reference}

Assistant's summary: 
{prediction}

Rating: """


def parse_res(res):
    match = re.search(r'\[\[(\d+)\]\]', res)
    if match:
        return int(match.group(1))
    else:
        return 1


def qa_rating(question, prediction, reference):
    
    def get_res(prompt):
        return call_api(
            msg=[{"role": "user", "content": prompt}],
            model_str="gpt-4o-2024-08-06",
            max_return_tokens=10,
            temperature=0.1,
        )
    
    prompt = QA_RATING_PROMPT.format(question=question, prediction=prediction, reference=reference)
    score = parse_res(get_res(prompt))
    return score


def sum_rating(question, prediction, reference):
    def get_res(prompt):
        return call_api(
            msg=[{"role": "user", "content": prompt}],
            model_str="gpt-4o-2024-08-06",
            max_return_tokens=10,
            temperature=0.1,
        )
    
    prompt = SUMMARIZATION_RATING_PROMPT.format(prediction=prediction, reference=reference)
    score = parse_res(get_res(prompt))
    return score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--res_dir", type=str, default="", help="The directory with a number of files, each with lines of prediction results")
    parser.add_argument("--output_dir", type=str, default="", help="The directory to output rating results")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    subname2score = {}
    for filename in os.listdir(args.res_dir):
        res_list = []
        subset_name = filename.split('.')[0]

        llm_rating = qa_rating
        highest_score = 3
        if subset_name in ["gov_report", "vcsum", "multinews", "qmsum"]:
            llm_rating = sum_rating
            highest_score = 5

        with open(os.path.join(args.res_dir, filename)) as f:
            for line in f.readlines():
                example = json.loads(line)
                res_list.append(example)

        score_list = []
        detail = {example["id"]: example for example in res_list}

        def run(example):
            qid = example["id"]
            question = example['question']
            pred_answer = example['pred']
            gold_answers = example['answers']

            max_score = 1
            for ref in gold_answers[:3]:
                s = llm_rating(question, pred_answer, ref)
                if s == highest_score:
                    max_score = s
                    break
                if s > max_score:
                    max_score = s

            return qid, max_score

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run, data) for data in detail.values()]

        for job in tqdm(as_completed(futures)):
            qid, score = job.result(timeout=None)
            detail[qid].update({"score": score})
            score_list.append(score)

        s_mean = (np.mean(score_list).item() - 1) / (highest_score - 1)
        subname2score[subset_name] = s_mean

        with open(os.path.join(args.output_dir, f"{subset_name}-details.json"), 'w') as f:
            json.dump(detail, f, indent=4, ensure_ascii=False)

    with open(os.path.join(args.output_dir, f"all-ratings.json"), 'w') as f:
        json.dump(subname2score, f, indent=4)
