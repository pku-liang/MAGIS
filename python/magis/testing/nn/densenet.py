from magis.op_graph import OpGraph


def _dense_layer(G: OpGraph, xs: list, growth_rate: int, bn_size: int):
    x = G.concat(xs, 1)
    # ignore BN in our evaluation for MAGIS and baselines
    x = G.relu(x)
    x = G.Conv2d(x, bn_size * growth_rate, kernel_size=1)
    # ignore BN in our evaluation for MAGIS and baselines
    x = G.relu(x)
    x = G.Conv2d(x, growth_rate, kernel_size=3, padding="same")
    return x


def _dense_block(G: OpGraph, x, num_layers: int, bn_size: int, growth_rate: int):
    xs = [x]
    for i in range(num_layers):
        x = _dense_layer(G, xs, growth_rate, bn_size)
        xs.append(x)
    return G.concat(xs, 1)


def _trans_block(G: OpGraph, x, num_output_features: int):
    # ignore BN in our evaluation for MAGIS and baselines
    x = G.Conv2d(G.relu(x), num_output_features, kernel_size=1)
    x = G.avg_pool2d(x, kernel_size=2, stride=2)
    return x


def _densenet(
    growth_rate: int = 32,
    block_config: tuple = (6, 12, 24, 16),
    num_init_features: int = 64,
    bn_size: int = 4,
):
    def _f(batch_size=1, in_channel_size=3, image_size=224, num_classes=1000):
        G = OpGraph()
        x = G.placeholder([batch_size, in_channel_size, image_size, image_size], "x")
        x = G.Conv2d(x, num_init_features, kernel_size=7, stride=2, padding=3)
        # ignore BN in our evaluation for MAGIS and baselines
        x = G.relu(x)
        x = G.avg_pool2d(x, kernel_size=3, stride=2, padding=1)

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            x = _dense_block(G, x, num_layers, bn_size, growth_rate)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                num_features //= 2
                x = _trans_block(G, x, num_features)

        # ignore BN in our evaluation for MAGIS and baselines
        x = G.relu(x)
        x = G.avg_pool2d(x, kernel_size=G[x].out_shape[-2])
        x = G.Conv2d(x, num_classes, kernel_size=1)
        return G, x

    return _f


densenet121 = _densenet(32, (6, 12, 24, 16), 64)
densenet161 = _densenet(48, (6, 12, 36, 24), 96)
densenet169 = _densenet(32, (6, 12, 32, 32), 64)
densenet201 = _densenet(32, (6, 12, 48, 32), 64)
