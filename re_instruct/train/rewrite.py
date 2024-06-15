"""finetune a rewrite llm"""

from functools import partial
from typing import Dict

import fire
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from re_instruct.data_collator import DataCollator
from re_instruct.utils import read_json
from re_instruct.constants import (
    REWRITE_PROMPT,
    WEB_TEXT_HEAD,
    QUESTION_HEAD,
    ANSWER_HEAD,
)


def tokenize_data_point(
    data_point: Dict[str, str],
    tokenizer: AutoTokenizer,
    model_max_length: int,
):
    """tokenize individual data point"""
    # get prompt str and full str
    prompt_str = (
        REWRITE_PROMPT
        + WEB_TEXT_HEAD
        + data_point["web_text"]
        + QUESTION_HEAD
        + data_point["instruction"]
        + ANSWER_HEAD
    )
    full_str = prompt_str + data_point["output"] + tokenizer.eos_token

    # tokenize full str
    tokenized_full_str = tokenizer(full_str)

    # set up labels
    prompt_len = len(tokenizer(prompt_str).input_ids)
    tokenized_full_str["labels"] = [-100] * prompt_len + tokenized_full_str[
        "input_ids"
    ][prompt_len:]

    # manually truncate to model_max_length
    tokenized_full_str["input_ids"] = tokenized_full_str["input_ids"][:model_max_length]
    tokenized_full_str["attention_mask"] = tokenized_full_str["attention_mask"][
        :model_max_length
    ]
    tokenized_full_str["labels"] = tokenized_full_str["labels"][:model_max_length]

    return tokenized_full_str


def main(
    data_path: str,
    model_name_or_path: str,
    model_max_length: int,
    padding: str = "max_length",
    pad_to_multiple_of: int = 8,
    return_tensors: str = "pt",
    **kwargs,
):
    """main entry point"""
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
    )

    # read data
    data = read_json(data_path)

    # get dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(
        partial(
            tokenize_data_point,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
        ),
        remove_columns=list(dataset.features),
        num_proc=8,
    )

    # get data_collator
    data_collator = DataCollator(
        tokenizer=tokenizer,
        padding=padding,
        max_length=model_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
    )

    # get training_args
    training_args = TrainingArguments(**kwargs)

    # get trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # start training
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
