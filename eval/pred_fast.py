import os
import re
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
from nltk.tokenize import PunktSentenceTokenizer

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="", help="The model directory")
    parser.add_argument('--data_dir', type=str, default=".", help="The directory with input files")
    parser.add_argument('--max_length', type=int, default=64000, help="Maximum input tokens in the prompt")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--output_dir', type=str, default=".", help="The output directory")
    parser.add_argument('--add_marks_to_text', action='store_true', help="Whether add number marks to document chunks")
    return parser.parse_args(args)


def add_marks_to_document(document):
    def text_split_by_punctuation(original_text, return_dict=True):
        # text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', original_text)  # separate period without space
        text = original_text
        custom_sent_tokenizer = PunktSentenceTokenizer(text)
        punctuations = r"([。；！？])"  # For Chinese support

        separated = custom_sent_tokenizer.tokenize(text)
        separated = sum([re.split(punctuations, s) for s in separated], [])
        # Put the punctuations back to the sentence
        for i in range(1, len(separated)):
            if re.match(punctuations, separated[i]):
                separated[i-1] += separated[i]
                separated[i] = ''

        separated = [s for s in separated if s != ""]
        if len(separated) == 1:
            separated = original_text.split('\n\n')
        separated = [s.strip() for s in separated if s.strip() != ""]

        if not return_dict:
            return separated
        else:
            pos = 0
            res = []
            for i, sent in enumerate(separated):
                st = original_text.find(sent, pos)
                assert st != -1, sent
                ed = st + len(sent)
                res.append(
                    {
                        'c_idx': i,
                        'content': sent,
                        'start_idx': st,
                        'end_idx': ed,
                    }
                )
                pos = ed
            return res
        
    all_c_sents = text_split_by_punctuation(document)
    numbered = ""
    for i, c in enumerate(all_c_sents):
        st, ed = c['start_idx'], c['end_idx']
        assert c['content'] == document[st:ed], c
        ed = all_c_sents[i + 1]['start_idx'] if i < len(all_c_sents) - 1 else len(document)
        numbered += f"<C{i}>" + document[st:ed]

    return numbered


# This is the customized building prompt for chat models
def build_chat(example, tokenizer, add_marks_to_text):
    template = """You are given a long document and a question. You are required to answer this question based on the document.
Please first output your thinking process wrapped by <think> and </think>, and then output your final answer wrapped by <answer> and </answer>.
Format the output as: <think>YOUR THINKING PROCESS</think><answer>YOUR FINAL ANSWER</answer>

<document>
{doc_str}
</document>

<question>{question}</question>"""

    doc_str = example["context"]
    if add_marks_to_text:
        doc_str = add_marks_to_document(doc_str)
    question = example["input"]

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": template.format(doc_str=doc_str, question=question)}], 
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model, tokenizer, add_marks_to_text, out_path, lock):
    device = torch.device(f'cuda:{rank}')

    for json_obj in tqdm(data):
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            if dataset == "gov_report":
                json_obj["input"] = "What's this document about? Write a one-page summary of this document."
            if dataset == "vcsum":
                json_obj["input"] = "这次会议的主要内容是什么？请写一个会议纪要。"

            prompt = build_chat(json_obj, tokenizer, add_marks_to_text)

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                # num_beams=1,
                do_sample=True,
                temperature=0.6
            )[0]
        res = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        res = re.sub(r"\n{2,}", "\n", res)
        
        pred = re.search(r"<answer>(.+?)</answer>", res, re.DOTALL)
        if not pred:
            pred = re.search(r"<answer>(.+?)$", res, re.DOTALL)
        if not pred:
            pred = re.search(r"</think>(.+?)$", res, re.DOTALL)

        if pred:
            pred = pred.group(1)
        else:
            pred = res

        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "llm_res": res, "answers": json_obj["answers"], "question": json_obj["input"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "id": json_obj["_id"]}, f, ensure_ascii=False)
                f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model
    max_length = args.max_length
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", 
                    "dureader", "gov_report", "qmsum", "vcsum"]
        # datasets = ["gov_report", "vcsum"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2cot_prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists(f"{args.output_dir}"):
        os.makedirs(f"{args.output_dir}")
    if not os.path.exists(f"{args.output_dir}"):
        os.makedirs(f"{args.output_dir}")

    models, tokenizers = [], []
    for rank in range(world_size):
        model, tokenizer = load_model_and_tokenizer(args.model_path, f"cuda:{rank}")
        models.append(model)
        tokenizers.append(tokenizer)

    for dataset in datasets:
        if args.e:
            data = load_dataset("json", data_files={"test": f"{args.data_dir}/{dataset}_e.jsonl"})["test"]
            if not os.path.exists(f"{args.output_dir}/pred_e"):
                os.makedirs(f"{args.output_dir}/pred_e/pred_e")
            out_path = f"{args.output_dir}/pred_e/{dataset}.jsonl"
        else:
            data = load_dataset("json", data_files={"test": f"{args.data_dir}/{dataset}.jsonl"})["test"]
            if not os.path.exists(f"{args.output_dir}/pred"):
                os.makedirs(f"{args.output_dir}/pred")
            out_path = f"{args.output_dir}/pred/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        lock = mp.Lock()
        processes = []
        for rank in range(world_size):
            model = models[rank]
            tokenizer = tokenizers[rank]
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, 
                        max_gen, prompt_format, dataset, device, model, tokenizer, args.add_marks_to_text, out_path, lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
