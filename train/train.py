# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl.trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from rl_utils.reward import completely_two_ways_reward
from custom_trainer import DynamicSamplerTrainer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_model_name_or_path (`str` or `None`):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
    """

    train_data_path: str = field(
        default="",
        metadata={"help": "Path to training data."},
    )

    eval_data_path: str = field(
        default="",
        metadata={"help": "Path to evaluation data."},
    )

    min_lr: float = field(
        default=0,
        metadata={"help": "Minimum learning rate."}
    )

    start_pA: float = field(
        default=1.0,
        metadata={"help": "Initial probability of sampling from dataset A."}
    )

    end_pA: float = field(
        default=0.0,
        metadata={"help": "Final probability of sampling from dataset A."}
    )

    sampling_schedule: str = field(
        default="linear",
        metadata={"help": "Schedule for pA: linear, exponential, cosine."}
    )


def main(script_args, training_args, model_args):
    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    # Load the dataset
    dataset = load_dataset("json", data_files={"train": script_args.train_data_path, "test": script_args.eval_data_path})

    def preprocess(example):

        template = """You are given a long document and a question. You are required to answer this question based on the document.
Please first output your thinking process wrapped by <think> and </think>, and then output your final answer wrapped by <answer> and </answer>.
Format the output as: <think>YOUR THINKING PROCESS</think><answer>YOUR FINAL ANSWER</answer>

<document>
{doc_str}
</document>

<question>{question}</question>
"""

        question = example["input"]
        doc_str = example["context"]

        prompt = [{"role": "user", "content": template.format(doc_str=doc_str, question=question)}]

        return {
            "prompt": prompt,
            "datatype": example.get("datatype", "original-musique")
        }
    
    train_dataset = dataset[script_args.dataset_train_split].map(
        preprocess,
        num_proc=8,
        remove_columns=None,
        load_from_cache_file=False,
        desc="Running tokenizer on training dataset",
    )

    eval_dataset = dataset[script_args.dataset_test_split].map(
        preprocess,
        num_proc=8,
        remove_columns=None,
        load_from_cache_file=False,
        desc="Running tokenizer on evaluation dataset",
    ) if training_args.eval_strategy != "no" else None

    def lr_lambda(current_step: int):
        # total_steps 可以从 Trainer 的 args 获取
        if current_step > training_args.max_steps:
            return script_args.min_lr / training_args.learning_rate
        # 线性衰减到 min_lr
        decay_factor = 1 - (current_step / training_args.max_steps)
        return (script_args.min_lr + (training_args.learning_rate - script_args.min_lr) * decay_factor) / training_args.learning_rate

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    training_args.start_pA = script_args.start_pA
    training_args.end_pA = script_args.end_pA
    training_args.sampling_schedule = script_args.sampling_schedule

    # Initialize the GRPO trainer
    trainer = DynamicSamplerTrainer(
        model=model,
        reward_funcs=[
            completely_two_ways_reward
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    sys.exit(0)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)