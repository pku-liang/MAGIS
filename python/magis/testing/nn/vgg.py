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
            dilation=dilation,
        )
    return _f

def _conv_block(
    inplanes: int,
    outplanes: int,
    n: int
):
    def _f(G: OpGraph, x):
        cur_in = inplanes
        for i in range(n):
            x = _conv3x3(cur_in, outplanes)(G, x)
            cur_in = inplanes
        return x
    return _f

def _vgg():
    def _f(batch_size=1, in_channel_size=3, image_size=224, num_classes=1000):
        G = OpGraph()
        x = G.placeholder([batch_size, in_channel_size, image_size, image_size], "x")
        x = _conv_block(3, 64, 2)(G, x)
        x = G.avg_pool2d(x, 2, stride=2)
        x = _conv_block(64, 128, 2)(G, x)
        x = G.avg_pool2d(x, 2, stride=2)
        x = _conv_block(128, 256, 3)(G, x)
        x = G.avg_pool2d(x, 2, stride=2)
        x = _conv_block(256, 512, 3)(G, x)
        x = G.avg_pool2d(x, 2, stride=2)
        x = _conv_block(512, 512, 3)(G, x)
        x = G.avg_pool2d(x, 2, stride=2)
        x = G.reshape(x, [batch_size, 25088])
        x = G.Linear(x, 4096)
        x = G.Linear(x, 4096)
        x = G.Linear(x, num_classes)
        return G, x

    return _f

vgg16 = _vgg()
