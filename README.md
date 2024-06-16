# REInstruct

This is the official repository for ACL 2024 Findings paper "REInstruct: Building Instruction Data from Unlabeled Corpus".

## Installation

```
# create conda environment
conda create -n re_instruct python=3.8
conda activate re_instruct

# editable install
pip install -e .

# install flash attention independently
pip install flash-attn==2.5.7
```

## Prepare Unlabeled Texts

Download [C4 dataset](https://huggingface.co/datasets/allenai/c4) and decompress text files in `en` folder. Select candidate texts using the following scripts:

```bash
python re_instruct/data/prepare_unlabeled_texts.py \
    --c4_en_dir <path_to_c4_en_folder> \
    --output_dir <output_dir>
```

## Filter Rewritten Responses

To filter out failed rewritten responses:

```bash
python re_instruct/data/filter_rewritten.py \
    --data_path <path_to_data_for_filtering> \
    --output_dir <output_dir> \
    --remove_refusal True
```

## Sunburst Visualization

[Installing required dependency](https://github.com/nikitakit/self-attentive-parser#installation) and visualize instructions using the following scripts:

```bash
python re_instruct/data/sunburst_visualize.py \
    --data_path example.json \
    --output_svg_path example.svg
```
