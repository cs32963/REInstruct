"""report basic data info"""

import fire
import numpy as np

from re_instruct.utils import read_json, prettify_dict


def sft_data(
    data_path: str,
):
    """report basic statistics of sft data"""
    # read data
    data = read_json(data_path)
    # get insts and outputs
    insts = [_["instruction"] for _ in data]
    outputs = [_["output"] for _ in data]

    # get info
    info = {}
    info["data_size"] = len(data)
    info["avg_inst_len"] = round(np.array(list(map(len, insts))).mean(), 2)
    info["avg_output_len"] = round(np.array(list(map(len, outputs))).mean(), 2)

    # print info
    print(prettify_dict(info))


if __name__ == "__main__":
    fire.Fire()
