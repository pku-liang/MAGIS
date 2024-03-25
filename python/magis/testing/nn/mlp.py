from magis.op_graph import OpGraph


def simple_mlp_net(batch_size, hidden_sizes: list, num_classes=100, activations="relu"):
    if not isinstance(activations, (list, tuple)):
        activations = [activations] * (len(hidden_sizes) - 1)
    graph = OpGraph()
    x = graph.placeholder([batch_size, hidden_sizes[0]], name_hint="x")
    for ih, oh, act in zip(hidden_sizes, hidden_sizes[1:], activations):
        w = graph.placeholder([ih, oh], name_hint="w", is_weight=True)
        x = graph.matmul(x, w)
        if act is not None:
            x = graph.ewise_uniop(act, x)
    w = graph.placeholder(
        [hidden_sizes[-1], num_classes], name_hint="w", is_weight=True
    )
    y = graph.matmul(x, w)
    return graph, y
