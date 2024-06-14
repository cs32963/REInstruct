"""data collator"""

from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass
class DataCollator:
    """Data collator that will dynamically pad input, as well as labels"""

    tokenizer: AutoTokenizer
    padding: str
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"

    def __call__(self, features):
        # get input_ids
        input_ids = [feature["input_ids"] for feature in features]

        # get pad_to_len
        if self.padding == "longest":
            pad_to_len = max(len(_) for _ in input_ids)
            if self.pad_to_multiple_of is not None:
                pad_to_len = (
                    (pad_to_len + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
        elif self.padding == "max_length":
            pad_to_len = self.max_length

        # get labels
        if "labels" in features[0]:
            labels = [feature["labels"] for feature in features]
        else:
            labels = None

        # get padding_side
        padding_side = self.tokenizer.padding_side

        # pad labels
        if labels is not None:
            for feature in features:
                remainder = [-100] * (pad_to_len - len(feature["labels"]))
                if padding_side == "left":
                    feature["labels"] = remainder + feature["labels"]
                elif padding_side == "right":
                    feature["labels"] = feature["labels"] + remainder

        # pad input_ids and attention_mask
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        return features
