from magis.op_graph import OpGraph


def unet(
    batch_size=1,
    image_size=256,
    in_channel_size=1,
    out_channel_size=1,
    hidden_sizes=[64, 128, 256, 512, 1024],
):
    graph = OpGraph()
    x = graph.placeholder((batch_size, in_channel_size, image_size, image_size), "x")
    res_convs = []

    # down layers
    for h in hidden_sizes[:-1]:
        x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
        x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
        res_convs.append(x)
        x = graph.avg_pool2d(x, 2, stride=2)
    # middle layers
    h = hidden_sizes[-1]
    x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
    x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
    # up layers
    for i, h in reversed(list(enumerate(hidden_sizes[:-1]))):
        x = graph.interpolate(x, scale=2)
        x = graph.concat([res_convs[i], x], dim=1)
        x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
        x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
    x = graph.Conv2d(x, 2, 3, activation="relu", padding="same")
    x = graph.Conv2d(x, out_channel_size, 1, activation="sigmoid")

    return graph, x


def unet2p(
    batch_size=1,
    image_size=256,
    in_channel_size=1,
    out_channel_size=1,
    hidden_sizes=[64, 128, 256, 512, 1024],
    using_deep_supervision=False,
):
    graph = OpGraph()
    x = graph.placeholder((batch_size, in_channel_size, image_size, image_size), "x")
    res_conv = [[None for _ in hidden_sizes] for _ in hidden_sizes]

    for i, h in enumerate(hidden_sizes):
        x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
        res_conv[i][0] = x
        x = graph.avg_pool2d(x, 2, stride=2)

        for j, hh in enumerate(reversed(hidden_sizes[:i])):
            up = res_conv[i][j]
            up = graph.interpolate(up, scale=2)
            cats = reversed([res_conv[i - 1 - off][j - off] for off in range(j + 1)])
            cat = graph.concat([up, *cats], dim=1)
            conv = graph.Conv2d(cat, hh, 3, padding="same", activation="relu")
            res_conv[i][j + 1] = conv

    if using_deep_supervision:
        convs = []
        for i in len(hidden_sizes):
            conv = res_conv[i][i]
            convs.append(graph.Conv2d(conv, out_channel_size, 1, activation="sigmoid"))
            return graph, convs
    else:
        conv = res_conv[-1][-1]
        conv = graph.Conv2d(conv, out_channel_size, 1, activation="sigmoid")
        return graph, conv


def unet3p(
    batch_size=1,
    image_size=256,
    in_channel_size=1,
    out_channel_size=1,
    hidden_sizes=[64, 128, 256, 512, 1024],
):
    graph = OpGraph()
    x = graph.placeholder((batch_size, in_channel_size, image_size, image_size), "x")
    res_convs = []

    # down layers
    for h in hidden_sizes[:-1]:
        x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
        x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
        res_convs.append(x)
        x = graph.avg_pool2d(x, 2, stride=2)
    # middle layers
    h = hidden_sizes[-1]
    x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
    x = graph.Conv2d(x, h, 3, padding="same", activation="relu")
    res_convs.append(x)
    # up layers
    for i, h in reversed(list(enumerate(hidden_sizes[:-1]))):
        cats = []
        for j, rc in reversed(list(enumerate(res_convs))):
            if j > i:
                rc = graph.interpolate(rc, scale=2 ** (j - i))
            elif j < i:
                rc = graph.avg_pool2d(rc, 2 ** (i - j), stride=2 ** (i - j))
            rc = graph.Conv2d(rc, hidden_sizes[0], 3, padding="same", activation="relu")
            cats.append(rc)
        cat = graph.concat(cats, dim=1)
        res_convs[i] = cat
    out = res_convs[0]
    out = graph.Conv2d(
        out,
        hidden_sizes[0] * len(hidden_sizes),
        3,
        padding="same",
        activation="sigmoid",
    )
    out = graph.Conv2d(out, out_channel_size, 3, padding="same", activation="sigmoid")

    return graph, out
