from magis.op_graph import OpGraph


def _conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
):
    assert groups == 1
    def _f(G: OpGraph, x):
        return G.Conv2d(
            x,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            # groups=groups,
            dilation=dilation,
        )

    return _f


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    def _f(G: OpGraph, x):
        return G.Conv2d(x, out_planes, kernel_size=1, stride=stride)

    return _f


def _basic_block(
    G: OpGraph,
    x,
    inplanes: int,
    planes: int,
    stride: int = 1,
    downsample=None,
    groups: int = 1,
    base_width: int = 64,
    dilation: int = 1,
):
    y = _conv3x3(inplanes, planes, stride)(G, x)
    # ignore BN in our evaluation for MAGIS and baselines
    y = G.relu(y)

    y = _conv3x3(planes, planes)(G, y)
    # ignore BN in our evaluation for MAGIS and baselines
    if downsample:
        x = downsample(G, x)
    y = G.add(y, x)
    y = G.relu(y)

    return y


_basic_block.expansion = 1


def _bottleneck(
    G: OpGraph,
    x,
    inplanes: int,
    planes: int,
    stride: int = 1,
    downsample=None,
    groups: int = 1,
    base_width: int = 64,
    dilation: int = 1,
):
    width = int(planes * (base_width / 64.0)) * groups
    y = _conv1x1(inplanes, width)(G, x)
    # ignore BN in our evaluation for MAGIS and baselines
    y = G.relu(y)

    y = _conv3x3(width, width, stride, groups, dilation)(G, y)
    # ignore BN in our evaluation for MAGIS and baselines
    y = G.relu(y)

    y = _conv1x1(width, planes * _bottleneck.expansion)(G, y)
    # ignore BN in our evaluation for MAGIS and baselines
    if downsample is not None:
        x = downsample(G, x)
    y = G.add(y, x)
    y = G.relu(y)

    return y


_bottleneck.expansion = 4


def _resnet(
    block,
    layers: list,
    groups: int = 1,
    width_per_group: int = 64,
):
    def _f(batch_size=1, in_channel_size=3, image_size=224, num_classes=1000):
        G = OpGraph()
        x = G.placeholder([batch_size, in_channel_size, image_size, image_size], "x")
        inplanes = 64
        dilation = 1
        base_width = width_per_group

        x = G.Conv2d(x, inplanes, kernel_size=7, stride=2, padding=3)
        # ignore BN in our evaluation for MAGIS and baselines
        x = G.relu(x)
        x = G.avg_pool2d(x, kernel_size=3, stride=2, padding=1)

        def _make_layer(x, planes: int, blocks: int, stride: int = 1):
            nonlocal inplanes
            downsample = None
            if stride != 1 or inplanes != planes * block.expansion:

                def downsample(G: OpGraph, x):
                    x = _conv1x1(inplanes, planes * block.expansion, stride)(G, x)
                    # ignore BN in our evaluation for MAGIS and baselines
                    return x

            x = block(G, x, inplanes, planes, stride, downsample, groups, base_width)
            inplanes = planes * block.expansion
            for _ in range(1, blocks):
                x = block(
                    G,
                    x,
                    inplanes,
                    planes,
                    groups=groups,
                    base_width=base_width,
                    dilation=dilation,
                )

            return x

        x = _make_layer(x, 64, layers[0])
        x = _make_layer(x, 128, layers[1], stride=2)
        x = _make_layer(x, 256, layers[2], stride=2)
        x = _make_layer(x, 512, layers[3], stride=2)
        x = G.avg_pool2d(x, kernel_size=G[x].out_shape[-2])
        x = G.Conv2d(x, num_classes, kernel_size=1)

        return G, x

    return _f


resnet34 = _resnet(_basic_block, [3, 4, 6, 3])
resnet50 = _resnet(_bottleneck, [3, 4, 6, 3])
resnet101 = _resnet(_bottleneck, [3, 4, 23, 3])
resnet152 = _resnet(_bottleneck, [3, 8, 36, 3])
# resnext50_32x4d = _resnet(_bottleneck, [3, 4, 6, 3], 32, 4)
# resnext101_32x8d = _resnet(_bottleneck, [3, 4, 23, 3], 32, 8)
# resnext101_64x4d = _resnet(_bottleneck, [3, 4, 23, 3], 64, 4)
wide_resnet50_2 = _resnet(_bottleneck, [3, 4, 6, 3], width_per_group=64 * 2)
wide_resnet101_2 = _resnet(_bottleneck, [3, 4, 23, 3], width_per_group=64 * 2)
