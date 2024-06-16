"""filter rewritten data"""

from pathlib import Path
from typing import Dict

import fire

from re_instruct.utils import read_json, save_json


def rewritten_filter(
    data_path: str,
    output_dir: str,
    remove_refusal: bool = False,
):
    """minimal filter"""

    # helper function
    def keep(data_point: Dict[str, str]):
        """whether to keep the data_point"""
        # get inst and output
        inst = data_point["instruction"]
        output = data_point["output"]

        # inst too long
        if len(inst) > 500:
            return False

        # contains too much '- '
        if output.count("- ") > 10:
            return False

        # contains 'this' in inst
        if "this" in inst.lower():
            return False

        # contains 'web text' in output
        if "web text" in output.lower():
            return False

        # contains 'based on the information provided' in output
        if "based on the information provided" in output.lower():
            return False

        if remove_refusal:
            # contains 'sorry' in output
            if "sorry" in output.lower():
                return False

            # contains 'i apologize' in output
            if "i apologize" in output.lower():
                return False

        return True

    # read data
    data = read_json(data_path)
    filtered_data = [_ for _ in data if keep(_)]

    # save filtered data
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    save_json(filtered_data, Path(output_dir) / "data.json")


if __name__ == "__main__":
    fire.Fire(rewritten_filter)
