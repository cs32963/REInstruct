"""visualize verb-noun structure of instructions"""

import string

import fire
import spacy

# NB: this import line is required for add pipe benepar
import benepar
from tqdm import tqdm
import pandas as pd
import plotly.express as px

from re_instruct.utils import read_json

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})


def main(
    data_path: str,
    output_svg_path: str,
):
    """visualize verb-noun structure of instructions"""
    # read data
    data = read_json(data_path)

    # get insts
    insts = [_["instruction"] for _ in data]

    # helper function to get verb-noun
    def get_verb_noun(inst):
        """get verb-noun in inst"""
        # 512 is the max supported len in spacy
        inst = inst[:512].strip()

        # FIXME: unkonwn error
        # seems to be some tokenizer error, remove all punctuations for a temporary fix
        for punc in string.punctuation:
            inst = inst.replace(punc, " ")
        inst = " ".join(inst.split())

        # get first sent
        doc = nlp(inst)
        first_sent = list(doc.sents)[0]

        # get verb-dobj
        def get_verb_dobj(root):
            """get verb and dobj"""
            # first check if the current node and its children satisfy the condition
            if root.pos_ == "VERB":
                for child in root.children:
                    if child.dep_ == "dobj" and child.pos_ == "NOUN":
                        return root.lemma_, child.lemma_
                return root.lemma_, None

            # if not, check its children
            for child in root.children:
                return get_verb_dobj(child)

            # if no children satisfy the condition, return None
            return None, None

        verb, dobj = get_verb_dobj(first_sent.root)
        ret = {
            "verb": verb,
            "noun": dobj,
        }
        return ret

    # verb_nouns = map(get_verb_noun, insts)
    verb_nouns = []
    for inst in tqdm(insts):
        verb_nouns.append(get_verb_noun(inst))

    # draw sunburst graph
    verb_nouns = pd.DataFrame(verb_nouns)
    phrases = pd.DataFrame(verb_nouns).dropna()
    phrases[["verb", "noun"]].groupby(["verb", "noun"]).size().sort_values(
        ascending=False
    )
    top_verbs = phrases[["verb"]].groupby(["verb"]).size().nlargest(20).reset_index()
    df = phrases[phrases["verb"].isin(top_verbs["verb"].tolist())]
    df = (
        df.groupby(["verb", "noun"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
    )
    df = (
        df.groupby("verb")
        .apply(lambda x: x.sort_values("count", ascending=False).head(4))
        .reset_index(drop=True)
    )
    # df = df[df['count'] > 30]
    fig = px.sunburst(df, path=["verb", "noun"], values="count")
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        font_family="Times New Roman",
    )

    # save as an svg file
    fig.write_image(output_svg_path)


if __name__ == "__main__":
    fire.Fire(main)
