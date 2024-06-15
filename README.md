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

## Sunburst Visualization

[Installing required dependency](https://github.com/nikitakit/self-attentive-parser#installation) and visualize instructions using the following scripts:

```bash
python re_instruct/data/sunburst_visualize.py \
    --data_path example.json \
    --output_svg_path example.svg
```
