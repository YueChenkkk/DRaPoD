
import re
import math
import random
from typing import Union, Any
import types
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed

from transformers import Trainer, DynamicCache
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
import datasets

from trl.trl import GRPOTrainer
from trl.trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.trl.extras.profiling import profiling_context, profiling_decorator
from trl.trl.import_utils import is_rich_available, is_vllm_available
from trl.trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation


class DynamicRepeatSampler(Sampler):
    def __init__(self,
                 data_source,
                 mini_repeat_count: int,
                 batch_size: int = 1,
                 repeat_count: int = 1,
                 seed: int = None,
                 start_pA=0.9,
                 end_pA=0.1,
                 total_steps=1000,
                 schedule="linear"):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

        self.start_pA = start_pA
        self.end_pA = end_pA
        self.total_steps = total_steps
        self.schedule = schedule

        self.indices_A = [i for i, d in enumerate(data_source) if d["difficulty"] in ["easy", "medium"]]
        self.indices_B = [i for i, d in enumerate(data_source) if d["difficulty"] in ["hard", "extreme"]]

        self.current_epoch = 0
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        print(f"rank:{torch.distributed.get_rank()}", self.current_epoch)

    def get_pA(self, global_step):
        ratio = min(global_step / max(1, self.total_steps - 1), 1.0)
        if self.schedule == "linear":
            return self.start_pA + ratio * (self.end_pA - self.start_pA)
        elif self.schedule == "exponential":
            decay_rate = (self.end_pA / self.start_pA) ** ratio
            return self.start_pA * decay_rate
        elif self.schedule == "cosine":
            cos_decay = 0.5 * (1 + math.cos(math.pi * ratio))
            return self.end_pA + (self.start_pA - self.end_pA) * cos_decay
        elif self.schedule == "staged":
            if ratio < 0.5:
                return self.start_pA
            else:
                return self.end_pA
        else:
            raise ValueError(f"未知调度类型: {self.schedule}")

    def __iter__(self):
        indices = []
        global_step_offset = self.current_epoch * self.num_samples

        # 固定顺序，确保多 rank 一致
        base_perm = torch.arange(self.num_samples).tolist()

        for local_step, _ in enumerate(base_perm):
            global_step = global_step_offset + local_step
            pA = self.get_pA(global_step)

            # 用 torch.randint 确保多 rank 同步
            if torch.rand(1, generator=self.generator).item() < pA and self.indices_A:
                idx = self.indices_A[torch.randint(len(self.indices_A), (1,), generator=self.generator).item()]
            else:
                idx = self.indices_B[torch.randint(len(self.indices_B), (1,), generator=self.generator).item()]

            indices.append(idx)

        # 按 batch repeat
        for i in range(0, len(indices), self.batch_size):
            chunk = indices[i:i + self.batch_size]
            if len(chunk) < self.batch_size:
                continue
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self):
        return self.num_samples * self.mini_repeat_count * self.repeat_count

class DynamicRepeatBatchSampler(BatchSampler):
    """
    包装 DynamicRepeatSampler，使其能作为 batch_sampler 传给 DataLoader。
    这样 Accelerate 会正确 shard，不会丢 sampler。
    """

    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler  # 保留底层 sampler

    def set_epoch(self, epoch: int):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

class DynamicSamplerTrainer(GRPOTrainer):
    def _get_train_sampler(self):
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        return DynamicRepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
            start_pA=self.args.start_pA,
            end_pA=self.args.end_pA,
            total_steps=self.args.max_steps * self.args.gradient_accumulation_steps * self.accelerator.num_processes * self.args.per_device_train_batch_size // self.num_generations,
            schedule=self.args.sampling_schedule,  # linear / exponential / cosine
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # 创建 batch_sampler 而不是 sampler
        base_sampler = self._get_train_sampler()
        batch_sampler = DynamicRepeatBatchSampler(
            base_sampler,
            batch_size=self._train_batch_size,
            drop_last=self.args.dataloader_drop_last,
        )

        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers
        }

        train_dataloader = DataLoader(train_dataset, **dataloader_params)
        train_dataloader = self.accelerator.prepare(train_dataloader)

        def batchsampler_set_epoch(self, epoch):
            # 调用内部 sampler 的 set_epoch
            if hasattr(self.batch_sampler, "set_epoch"):
                self.batch_sampler.set_epoch(epoch)

        # 动态绑定到 batch_sampler
        train_dataloader.batch_sampler.set_epoch = types.MethodType(batchsampler_set_epoch, train_dataloader.batch_sampler)

        return train_dataloader

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                    )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            # prompt_completion_lengths = [prompt_ids.size(1) + c.size(0) for c in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        prompt_completion_lengths = (torch.sum(completion_mask, dim=1) + prompt_ids.size(1)).tolist()  # length of prompt + completion without right pad

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = torch.zeros(prompt_ids.size(0), logits_to_keep, device=device)
                for i, (pc_len, pc) in enumerate(zip(prompt_completion_lengths, prompt_completion_ids)):
                    ref_logps = self._get_per_token_logps(
                        self.ref_model, pc[:pc_len].unsqueeze(0), None, pc_len - prompt_ids.size(1)
                    )
                    ref_per_token_logps[i][:pc_len - prompt_ids.size(1)] = ref_logps[0]
                # ref_per_token_logps = self._get_per_token_logps(
                #     self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                # )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        grouped_rewards = rewards.view(-1, self.num_generations)
        n_all_0 = n_all_1 = 0
        for i in range(grouped_rewards.size(0)):
            if torch.sum(grouped_rewards[i] > 0.5) == 0:
                n_all_0 += 1
            if torch.sum(grouped_rewards[i] > 0.5) == self.num_generations:
                n_all_1 += 1

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        self._metrics[mode]["all_0_ratio"].append(n_all_0)
        self._metrics[mode]["all_1_ratio"].append(n_all_1)

        # gather difficulty label
        difficulties = [x["difficulty"] for x in inputs]
        difficulties = gather_object(difficulties)
        hard_ratio = sum([1 if d in ["hard", "extreme"] else 0 for d in difficulties]) / len(difficulties)

        self._metrics[mode]["hard_ratio"].append(hard_ratio)

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "prompt_completion_lengths": prompt_completion_lengths
        }


