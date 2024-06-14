"""generate output"""

from pathlib import Path
from functools import partial
from typing import Dict

import fire
import torch
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

from re_instruct.data_collator import DataCollator
from re_instruct.utils import read_json, save_json
from re_instruct.constants import (
    VICUNA_V1_1_SYSTEM_MESSAGE,
    USER_HEAD,
    ASSISTANT_HEAD,
    SEED_PROMPT,
    AUG_PROMPT,
)


def tokenize_data_point(
    data_point: Dict[str, str],
    tokenizer: AutoTokenizer,
):
    """tokenize individual data point"""
    # get prompt
    prompt_str = (
        VICUNA_V1_1_SYSTEM_MESSAGE
        + USER_HEAD
        + data_point["instruction"]
        + ASSISTANT_HEAD
    )

    # tokenize prompt
    tokenized_prompt = tokenizer(prompt_str)

    return tokenized_prompt


@torch.no_grad()
def main(
    data_path: str,
    model_name_or_path: str,
    output_dir: str,
    prompt_type: str = "",  # seed, aug, seed_aug
    return_tensors: str = "pt",
    batch_size: int = 16,
    **kwargs,
):
    """main entry point"""
    # for multi-gpu inference
    accelerator = Accelerator()

    # make output_dir
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=False)

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    # set pad_token and padding_side
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # get model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": accelerator.process_index},
    )

    # read data
    data = read_json(data_path)
    if prompt_type == "seed":
        for d in data:
            d["original_instruction"] = d["instruction"]
            d["instruction"] += " " + SEED_PROMPT
    elif prompt_type == "aug_seed":
        for d in data:
            d["original_instruction"] = d["instruction"]
            d["instruction"] += " " + AUG_PROMPT
    elif prompt_type == "seed_aug":
        for d in data:
            d["original_instruction"] = d["instruction"]
            d["instruction"] += " " + SEED_PROMPT + " " + AUG_PROMPT

    # for each process
    with accelerator.split_between_processes(data) as data_subset:
        # get dataset
        dataset = Dataset.from_list(data_subset)
        dataset = dataset.map(
            partial(
                tokenize_data_point,
                tokenizer=tokenizer,
            ),
            remove_columns=list(dataset.features),
            num_proc=8,
        )

        # get data_collator
        data_collator = DataCollator(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors=return_tensors,
        )

        # get data_loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )

        # get generation_config
        generation_config = GenerationConfig.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

        # generate outputs
        outputs = []
        for batch in tqdm(data_loader):
            batch.to(device)
            model_out = model.generate(
                **batch,
                generation_config=generation_config,
            )
            raw_outputs = tokenizer.batch_decode(
                model_out[:, len(batch.input_ids[0]) :],
                skip_special_tokens=True,
            )
            outputs += [_.split(tokenizer.eos_token)[0] for _ in raw_outputs]

    # gather outputs
    gathered_outputs = gather_object(outputs)

    # post processing

    # add outputs to inst data
    for d, output in zip(data, gathered_outputs):
        d["output"] = output
        if prompt_type:
            d["running_inst"] = d["instruction"]
            d["instruction"] = d["original_instruction"]
            d.pop("original_instruction")

    # save results
    if accelerator.is_main_process:
        save_json(kwargs, Path(output_dir) / "args.json")
        save_json(data, Path(output_dir) / "data.json")


if __name__ == "__main__":
    fire.Fire(main)
