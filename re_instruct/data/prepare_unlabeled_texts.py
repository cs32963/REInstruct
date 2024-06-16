"""get candidate texts from c4"""

import re
import random
import string
from typing import Dict
from pathlib import Path

import fire
from tqdm import tqdm

from re_instruct.utils import (
    read_json,
    read_jsonl,
    save_json,
)


def main(
    c4_en_dir: str,
    output_dir: str,
    data_size: int = None,
):
    """get candidate text"""
    # make dir
    Path(output_dir).mkdir(parents=True, exist_ok=False)

    # get verbs
    verbs = read_json("data/verbs-all.json")
    norm_verb = set([_[0] for _ in verbs])
    ing_verb = set([_[-1] for _ in verbs])

    # helper function to keep text
    def keep(
        d: Dict[str, str],
    ):
        """whether to keep text or not"""
        # get text
        text = d["text"]
        text_lower = text.lower()

        # too short or too long
        if len(text) < 1200 or len(text) > 3000:
            return False

        # weird puncs
        weird_puncs = ["...", "…", "™", "#", "&", "*", "®", "@"]
        for weird_punc in weird_puncs:
            if weird_punc in text_lower:
                return False

        # contains personal pronouns
        forbidden_prons = [
            "we ",
            "our ",
            "i ",
            "i've ",
            "we've ",
            "we're ",
            "my ",
            "he ",
            "she ",
            "us ",
        ]
        cnt = 0
        for forbidden_pron in forbidden_prons:
            cnt += text_lower.count(" " + forbidden_pron) + text_lower.count(
                "\n" + forbidden_pron
            )
        if cnt > 2:
            return False

        # contains too many question marks
        if text_lower.count("?") > 1:
            return False

        # contains too many all capitalized words
        cnt = 0
        normed_text = text.replace("\n", " ")
        for punc in string.punctuation:
            normed_text.replace(punc, "")
        for normed_word in normed_text.split():
            if normed_word == normed_word.upper():
                cnt += 1
        if cnt > 2:
            return False

        # must contain some paragraph-starting verbs
        para_start_words = re.findall("\n([a-z]+) ", text_lower)
        # must contain various para_start_words
        if len(set(para_start_words)) < len(para_start_words) - 1:
            return False
        cnt1 = 0
        cnt2 = 0
        for para_start_word in para_start_words:
            if para_start_word in norm_verb:
                cnt1 += 1
            if para_start_word in ing_verb:
                cnt2 += 1
        all_para_cnt = text_lower.count("\n")
        if cnt1 >= 4 and cnt1 <= 10:
            if all_para_cnt < cnt1 + 2:
                return True

        if cnt2 >= 4 and cnt2 <= 10:
            if all_para_cnt < cnt2 + 2:
                return True

        return False

    # get candidate texts
    candidate_texts = []
    for file_path in tqdm(Path(c4_en_dir).iterdir()):
        if str(file_path).endswith(".json"):
            texts = read_jsonl(file_path)
            candidate_texts += [_ for _ in texts if keep(_)]

    # sample candidate texts
    if data_size is not None:
        candidate_texts = random.sample(candidate_texts, k=data_size)

    # post processing
    for d in candidate_texts:
        d["output"] = d["text"]
        d.pop("text")

    # save candidate texts
    save_json(candidate_texts, Path(output_dir) / "data.json")


if __name__ == "__main__":
    fire.Fire(main)
