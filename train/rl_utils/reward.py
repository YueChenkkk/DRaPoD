
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.tokenize import PunktSentenceTokenizer
from rank_bm25 import BM25Okapi
from .api import get_model, call_api, call_embedding_api
from transformers import AutoTokenizer

model_str = get_model()
sent_tokenizer = PunktSentenceTokenizer()
qwen_tokenizer = AutoTokenizer.from_pretrained("MODEL_PATH")

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def acc_format_reward(completions, answer, **kwargs) -> list[float]:
    format_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    
    scores = []
    for c, a in zip(completions, answer):
        format_match = re.match(format_pattern, c[0]["content"], re.DOTALL)
        if format_match:
            answer_match = re.search(r"<answer>(.*?)</answer>", c[0]["content"], re.DOTALL)
            pred_ans = answer_match.group(1)
            answer_pattern = re.escape(a.lower())
            is_correct = re.search(answer_pattern, pred_ans.lower())
            if is_correct:
                scores.append(1)
            else:
                scores.append(0.1)
        else:
            scores.append(0)

    return scores


def completely_two_ways_reward(completions, answer, input, datatype, **kwargs) -> list[float]:
    judge_template_majority_vote = """You are given a question about a document, the true answer to this question, and a candidate answer to this question. 
    
Your task is to first highlight the key point of the true answer, then verify whether the candidate answer has covered the same meaning as the true answer.

Please note that the candidate answer can be long and looks very different from the true answer, so \
you need to carefully analyze the candidate answer to determine if it has covered the essential aspects of the true answer.

Question: {question}

True answer: {answer}

Candidate answer: {pred_answer}

Please first highlight the key point in the true answer, then analyze whether the candidate answer has covered the same meaning in the true answer. Finally, output your judgement with "[[Yes]]" or "[[No]]".

Requirements:
- If the candidate answer has already covered the same meaning in the true answer, but contains some extra information, you should still answer "[[Yes]]", without judging the correctness of the extra information.
- If the candidate answer mentions any wrong information that contradicts the true answer, you should answer "[[No]]".
- If the candidate answer concludes the answer is not provided or cannot be determined, you should answer "[[No]]".
- For arithmetic questions, you should allow minor calculation mistakes (e.g., off by 1 year in age calculation), and still answer "[[Yes]]" if the candidate answer is very close to the true answer.
"""

    judge_template_simple = """You are given a question about a document, a candidate answer to this question, and the true answer to this question. You are required to judge whether the candidate answer is true.

Question: {question}

Candidate answer: {pred_answer}

True answer: {answer}

Does the candidate answer agree with the true answer? Just output "Yes" or "No".
"""

    format_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    
    scores = []
    for c, a, q, dt in zip(completions, answer, input, datatype):
        format_match = re.match(format_pattern, c[0]["content"], re.DOTALL)
        if format_match:
            answer_match = re.search(r"<answer>(.*?)</answer>", c[0]["content"], re.DOTALL)
            pred_ans = answer_match.group(1)
            answer_pattern = re.escape(a.lower())
            
            if dt == "original-musique":
                rule_correct = re.search(answer_pattern, pred_ans.lower()) is not None
                if rule_correct:
                    scores.append(1)
                    continue

                res = call_api(
                    msg=[{"role": "user", "content": judge_template_simple.format(question=q, pred_answer=pred_ans, answer=a)}],
                    model_str=model_str,
                    max_return_tokens=5,
                    temperature=0
                )

                llm_correct = re.search("Yes", res) is not None

                if llm_correct:
                    scores.append(1)
                else:
                    scores.append(0.1)
            else:
                rule_correct = pred_ans.lower().strip() == a.lower().strip()
                if rule_correct:
                    scores.append(1)
                    continue

                score_sum = 0
                for score_idx in range(5):
                    res = call_api(
                        msg=[{"role": "user", "content": judge_template_majority_vote.format(question=q, pred_answer=pred_ans, answer=a)}],
                        model_str=model_str,
                        max_return_tokens=500,
                        temperature=0.9
                    )

                    if re.search(r"\[\[Yes\]\]", res):
                        score_sum += 1

                    if score_sum == 3:
                        break

                    if score_idx + 1 - score_sum > 2:
                        break

                if score_sum >= 3:
                    scores.append(1)
                else:
                    scores.append(0.1)
        else:
            scores.append(0)

    return scores

