"""
UMAP comparison of entity embeddings: Word2Vec vs pretrained T5 (foundation-small).

Mixed file + subject nodes across /var/log/, /etc/, /usr/, /tmp/, /opt/ and /var/lib/
to test whether embeddings cluster by semantic function despite different path prefixes.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from types import SimpleNamespace

sys.path.insert(0, "/home/pids")

# ── Entity definitions: (node_type, path, category) ──────────────────────────

ENTITIES = [
    # /var/log/* — Logs
    ("file", "/var/log/auth.log",              "log"),
    ("file", "/var/log/syslog",                "log"),
    ("file", "/var/log/nginx/access.log",      "log_web"),
    ("file", "/var/log/apache2/error.log",     "log_web"),
    ("file", "/var/log/mysql/error.log",       "log_db"),
    ("file", "/var/log/postgresql/main.log",   "log_db"),
    # /etc/* — Configs + sysadmin
    ("file", "/etc/nginx/nginx.conf",          "config_web"),
    ("file", "/etc/apache2/apache2.conf",      "config_web"),
    ("file", "/etc/wireguard/wg0.conf",        "config_vpn"),
    ("file", "/etc/openvpn/server.conf",       "config_vpn"),
    ("file", "/etc/ssh/sshd_config",           "config"),
    ("file", "/etc/shadow",                    "sysadmin"),
    ("file", "/etc/sudoers",                   "sysadmin"),
    # /usr/[s]bin/* — four semantic classes
    ("subject", "/usr/sbin/nginx",             "webserver"),
    ("subject", "/usr/sbin/apache2",           "webserver"),
    ("subject", "/usr/sbin/sshd",              "ssh"),
    ("subject", "/usr/bin/ssh",                "ssh"),
    ("subject", "/usr/bin/curl",               "transfer"),
    ("subject", "/usr/bin/wget",               "transfer"),
    ("subject", "/usr/bin/mysql",              "db_client"),
    ("subject", "/usr/bin/psql",               "db_client"),
    ("subject", "/usr/sbin/mysqld",            "database"),
    # /tmp/* — tmp + archive
    ("file", "/tmp/sess_a1b2c3d4",             "tmp"),
    ("file", "/tmp/.X11-unix/X0",              "tmp"),
    ("file", "/tmp/build_output.tar.gz",       "archive"),
    ("file", "/tmp/firefox-installer.deb",     "archive"),
    # Singletons — cross-prefix behavioral traps
    ("subject", "/opt/nginx/sbin/nginx",       "webserver"),
    ("file",    "/var/lib/mysql/ibdata1",      "database_file"),
]

CATEGORY_COLORS = {
    # Logs — red family
    "log":           "#e6194b",
    "log_web":       "#f08080",
    "log_db":        "#8b0000",
    # Configs — orange family
    "config_web":    "#f58231",
    "config_vpn":    "#e6a817",
    "config":        "#c47a1a",
    # Sysadmin — brown
    "sysadmin":      "#7f4400",
    # Executables
    "webserver":     "#4363d8",
    "ssh":           "#911eb4",
    "transfer":      "#3cb44b",
    "db_client":     "#808000",
    "database":      "#556b00",
    # Tmp / archive
    "tmp":           "#aaaaaa",
    "archive":       "#000075",
    # Singleton
    "database_file": "#b5bd00",
}

W2V_PATH = "/home/artifacts/featurization/CADETS_E3/feat_training/bee9a2deaacaa8c8c4e7c87e35ca2f0c76c40318870d6f5fc5f3b318b50e86c1/stored_models/word2vec.model"
T5_DIR   = "/home/pids/weights/foundation-small"
T5_PT    = os.path.join(T5_DIR, "pretrain_mini_best.pt")
T5_TOK   = os.path.join(T5_DIR, "tokenizer.pt")

# ── Word2Vec embeddings ───────────────────────────────────────────────────────

def embed_word2vec(entities):
    from gensim.models import Word2Vec
    from nltk.tokenize import word_tokenize
    import re

    model = Word2Vec.load(W2V_PATH)
    emb_dim = model.vector_size
    zeros = np.zeros(emb_dim)
    decline_pct = 20

    def tokenize(label):
        label = re.sub(r"\\+", "/", label)
        return word_tokenize(label.replace("/", " / "))

    def cal_word_weight(n, pct):
        d = -1 / n * pct / 100
        a1 = 1 / n - 0.5 * (n - 1) * d
        return [a1 + i * d for i in range(n)]

    vecs = []
    for _, label, _ in entities:
        tokens = tokenize(label)
        weights = cal_word_weight(len(tokens), decline_pct)
        word_vecs = [model.wv[t] if t in model.wv else zeros for t in tokens]
        weighted = [w * v for w, v in zip(weights, word_vecs)]
        vec = np.mean(weighted, axis=0)
        norm = np.linalg.norm(vec)
        vecs.append(vec / (norm + 1e-12))
    return np.array(vecs, dtype=np.float32)


# ── T5 embeddings ─────────────────────────────────────────────────────────────

def embed_t5(entities):
    from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE
    from pidsmaker.spider.models.t5 import get_t5_config
    from transformers import T5EncoderModel

    cfg = SimpleNamespace(dataset=SimpleNamespace(name="CADETS_E3"))
    tokenizer = ProvenanceTokenizerBPE(cfg)
    tokenizer.load(T5_TOK)

    t5_config = get_t5_config(tokenizer.vocab_size, "mini", tokenizer.max_seq_len)
    encoder = T5EncoderModel(t5_config)

    sd_raw = torch.load(T5_PT, weights_only=True, map_location="cpu")
    sd_enc = {k[len("encoder."):]: v for k, v in sd_raw.items() if k.startswith("encoder.")}
    encoder.load_state_dict(sd_enc)
    encoder.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)

    vecs = []
    with torch.no_grad():
        for node_type, label, _ in entities:
            tids = tokenizer.tokenize_node(node_type, label)
            max_len = min(len(tids), tokenizer.max_seq_len)
            input_ids = torch.tensor([tids[:max_len]], dtype=torch.long).to(device)
            attn_mask = torch.ones(1, max_len, dtype=torch.long).to(device)

            out = encoder(input_ids=input_ids, attention_mask=attn_mask)
            hidden = out.last_hidden_state  # [1, L, H]
            mask_f = attn_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            vec = pooled[0].cpu().numpy()
            norm = np.linalg.norm(vec)
            vecs.append(vec / (norm + 1e-12))

    return np.array(vecs, dtype=np.float32)


# ── UMAP + plot ───────────────────────────────────────────────────────────────

def umap_2d(embeddings, n_neighbors=7, min_dist=0.2, random_state=42):
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state,
                        metric="cosine")
    return reducer.fit_transform(embeddings)


def plot_panel(ax, coords, entities, title):
    for (x, y), (_, label, cat) in zip(coords, entities):
        name = label.split("/")[-1] or label.split("/")[-2]
        color = CATEGORY_COLORS[cat]
        ax.scatter(x, y, color=color, s=120, zorder=3, edgecolors="white", linewidths=0.6)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5, color=color, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")


def main():
    print("Computing Word2Vec embeddings...")
    w2v_embs = embed_word2vec(ENTITIES)
    print(f"  shape: {w2v_embs.shape}")

    print("Computing T5 embeddings...")
    t5_embs = embed_t5(ENTITIES)
    print(f"  shape: {t5_embs.shape}")

    print("Running UMAP...")
    w2v_2d = umap_2d(w2v_embs)
    t5_2d  = umap_2d(t5_embs)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")

    plot_panel(axes[0], w2v_2d, ENTITIES, "Word2Vec (CADETS-E3)")
    plot_panel(axes[1], t5_2d,  ENTITIES, "T5 pretrained (foundation-small / mini)")

    patches = [
        mpatches.Patch(color=col, label=cat)
        for cat, col in CATEGORY_COLORS.items()
    ]
    fig.legend(handles=patches, loc="lower center", ncol=7,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    out = "/home/pids/umap_entities.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
