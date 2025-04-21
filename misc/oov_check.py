import os

import matplotlib.pyplot as plt
import wandb
from gensim.models import Word2Vec

from pidsmaker.config import *
from pidsmaker.utils.utils import *


def get_nid2scores(cfg):
    nid2scores = {}

    # load results dict
    results_dir = os.path.join(cfg.detection.evaluation._results_dir, "results.pth")
    results = torch.load(results_dir)

    for nid, data in results.items():
        nid2scores[int(nid)] = data["score"]

    return nid2scores


def get_nid2oov_component(cfg):
    feature_word2vec_model_path = (
        cfg.featurization.embed_nodes._model_dir + "feature_word2vec.model"
    )
    model = Word2Vec.load(feature_word2vec_model_path)

    indexid2msg = get_indexid2msg(cfg)

    nid2oov_component = {}
    for indexid, msg in indexid2msg.items():
        node_type, node_label = msg[0], msg[1]
        tokens = tokenize_label(node_label, node_type)
        seen_num = 0
        unseen_num = 0
        for token in tokens:
            if token in model.wv:
                seen_num += 1
            else:
                unseen_num += 1
        nid2oov_component[int(indexid)] = (seen_num, unseen_num)

    return nid2oov_component


def plot_oov_comp2scores(cfg):
    out_dir = os.path.join("../../results/featurization/", cfg.dataset.name)
    os.makedirs(out_dir, exist_ok=True)

    nid2scores = get_nid2scores(cfg)
    nid2comps = get_nid2oov_component(cfg)

    scores, oov_num, oov_percentage = [], [], []

    for nid, score in nid2scores.items():
        scores.append(score)
        oov_num.append(nid2comps[nid][1])
        oov_percentage.append((nid2comps[nid][1] / (nid2comps[nid][0] + nid2comps[nid][1])) * 100)

    # plotting score-OOV word number
    plt.figure(figsize=(8, 8))
    plt.scatter(oov_num, scores, color="blue")
    plt.xlabel("No. OOV words")
    plt.ylabel("Score")
    plt.show()
    plt.savefig(os.path.join(out_dir, "score_No_OOV_word.png"))
    print(f"Fig saved to {os.path.join(out_dir, 'score_No_OOV_word.png')}")
    plt.close()

    # plotting score-OOV word percentage
    plt.figure(figsize=(8, 8))
    plt.scatter(oov_percentage, scores, color="blue")
    plt.xlabel("OOV word percentage")
    plt.ylabel("Score")
    plt.show()
    plt.savefig(os.path.join(out_dir, "score_percentage_OOV_word.png"))
    print(f"Fig saved to {os.path.join(out_dir, 'score_percentage_OOV_word.png')}")
    plt.close()


def main(cfg):
    plot_oov_comp2scores(cfg)


if __name__ == "__main__":
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)

    exp_name = (
        args.exp
        if args.exp != ""
        else "|".join(
            [
                f"{k.split('.')[-1]}={v}"
                for k, v in args.__dict__.items()
                if "." in k and v is not None
            ]
        )
    )
    tags = args.tags.split(",") if args.tags != "" else [args.model]

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="featurization",
        name=exp_name,
        tags=tags,
    )

    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg)

    wandb.finish()
