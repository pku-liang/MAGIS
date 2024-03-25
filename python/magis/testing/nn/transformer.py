from magis.op_graph import OpGraph


def _transformer_block(G: OpGraph, x, emb_dim, ffn_dim, num_heads=1):
    y = G.LayerNorm(x, [G[x].out_shape[-1]])
    y = G.MultiheadAttention(y, y, y, emb_dim, num_heads)
    y = G.add(y, x)
    o = G.LayerNorm(y, [G[y].out_shape[-1]])
    o = G.Linear(o, ffn_dim)
    o = G.relu(o)  # use ReLU on behalf of activation functions
    o = G.Linear(o, emb_dim)
    o = G.add(o, y)
    return o


def _transformer_backbone(
    G: OpGraph, x, num_layers, hidden_size, num_heads, ffn_scale=4
):
    for _ in range(num_layers):
        x = _transformer_block(G, x, hidden_size, hidden_size * ffn_scale, num_heads)
    return x


def _language_transformer(
    batch_size=1,
    seq_len=512,
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    ffn_scale=4,
):
    # ignore pos embedding
    G = OpGraph()
    x = G.placeholder([batch_size, seq_len, hidden_size], "x")
    y = _transformer_backbone(G, x, num_layers, hidden_size, num_heads, ffn_scale)
    return G, y


def bert(name="base", batch_size=1, seq_len=512):
    configs = {
        "base": (12, 768, 12),
        "large": (24, 1024, 16),
        "medium": (8, 512, 8),
        "small": (4, 512, 8),
        "mini": (4, 256, 4),
        "tiny": (2, 128, 2),
    }
    num_layers, hidden_size, num_heads = configs[name]
    return _language_transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_scale=4,
    )


def gpt_neo(batch_size=1, seq_len=512):
    return _language_transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=24,
        hidden_size=2048,
        num_heads=16,
        ffn_scale=4,
    )


def btlm(batch_size=1, seq_len=512):
    return _language_transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=32,
        hidden_size=2560,
        num_heads=32,
        ffn_scale=4,
    )


def _vision_transformer(
    batch_size=1,
    in_channel_size=3,
    image_size=224,
    num_classes=1000,
    patch_size=16,
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    ffn_scale=4,
):
    # ignore pos embedding
    G = OpGraph()
    x = G.placeholder([batch_size, in_channel_size, image_size, image_size], "x")
    assert image_size % patch_size == 0
    num_patches = image_size // patch_size
    y = G.reshape(
        x,
        [batch_size, in_channel_size, num_patches, patch_size, num_patches, patch_size],
    )
    y = G.permute(y, [0, 2, 4, 1, 3, 5])
    y = G.reshape(
        y,
        [
            batch_size,
            num_patches * num_patches,
            in_channel_size * patch_size * patch_size,
        ],
    )
    y = _transformer_backbone(G, y, num_layers, hidden_size, num_heads, ffn_scale)
    y = G.Linear(y, num_classes)
    return G, y


def vit(name="base", batch_size=1, in_channel_size=3, image_size=224, num_classes=1000):
    configs = {
        "base": (12, 768, 12),
        "large": (24, 1024, 16),
        "huge": (32, 1280, 16),
        "small": (12, 384, 6),
        "tiny": (12, 192, 3),
    }
    if "-" in name:
        name, patch_size = name.split("-")
    else:
        patch_size = 16
    patch_size = int(patch_size)
    num_layers, hidden_size, num_heads = configs[name]
    return _vision_transformer(
        batch_size=batch_size,
        in_channel_size=in_channel_size,
        image_size=image_size,
        num_classes=num_classes,
        patch_size=patch_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_scale=4,
    )
