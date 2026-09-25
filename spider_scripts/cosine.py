#!/usr/bin/env python3
"""Print cosine similarity between two entity embeddings.

Usage:
    python cosine.py 'file /bin/bash' 'file /etc/passwd'
    python cosine.py 'subject nginx' 'subject apache2'
"""

import sys, os
sys.path.insert(0, '/home/pids')

import torch
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'weights', 'foundation-small')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_entity(s):
    """Parse 'type label' string into (node_type, label)."""
    parts = s.strip().split(None, 1)
    if len(parts) < 2:
        print(f"Error: expected 'type label', got: {s!r}", file=sys.stderr)
        sys.exit(1)
    return parts[0], parts[1]


def encode(model, tokenizer, node_type, label):
    """Encode a single entity and return its L2-normalized embedding."""
    ids = tokenizer.tokenize_node(node_type, label)
    if not ids:
        ids = [tokenizer.pad_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    with torch.no_grad():
        hidden = model.modified_fwd(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.full_like(input_ids, -100),
            skip_cls=True,
        )
    mask = attention_mask.unsqueeze(-1).float()
    emb = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    if norm > 1e-12:
        emb /= norm
    return emb


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip())
        sys.exit(1)

    type_a, label_a = parse_entity(sys.argv[1])
    type_b, label_b = parse_entity(sys.argv[2])

    from pidsmaker.spider.models.t5 import ProvenanceT5, get_t5_config
    from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE

    tokenizer = object.__new__(ProvenanceTokenizerBPE)
    tokenizer.load(f'{MODEL_DIR}/tokenizer.pt')

    state = torch.load(f'{MODEL_DIR}/pretrain_mini_best.pt', map_location='cpu', weights_only=False)
    config = get_t5_config(tokenizer.vocab_size, 'mini')
    model = ProvenanceT5(config)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()

    for name, ntype, label in [('A', type_a, label_a), ('B', type_b, label_b)]:
        ids = tokenizer.tokenize_node(ntype, label)
        tokens = [tokenizer.id2token.get(i, f'<unk:{i}>') for i in ids]
        print(f'  Entity {name}: [{ntype}] {label}')
        print(f'  Tokens {name}: {tokens}')

    emb_a = encode(model, tokenizer, type_a, label_a)
    emb_b = encode(model, tokenizer, type_b, label_b)

    similarity = float(np.dot(emb_a, emb_b))
    print(f'\nCosine similarity: {similarity:.4f}')


if __name__ == '__main__':
    main()
