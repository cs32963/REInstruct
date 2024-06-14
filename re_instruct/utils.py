"""utils"""

import json


def read_json(data_path):
    """read json data"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_jsonl(data_path):
    """read jsonl data"""
    with open(data_path, "r", encoding="utf-8") as f:
        json_lines = f.readlines()
    data = [json.loads(_) for _ in json_lines]
    return data


def save_json(data, save_path):
    """save json data"""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def prettify_dict(d):
    """prettify dict"""
    return json.dumps(d, indent=4)
