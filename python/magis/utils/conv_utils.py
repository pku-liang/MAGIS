
def get_conv1d_out_shape_and_pad(l, k, p, s=1, d=1):
    if p == "s":
        p2 = k + (k - 1) * (d - 1) - 1
        assert s == 1 and p2 % 2 == 0, (s, d, k)
        return l, p2 // 2
    else:
        return (l + 2 * p - k - (k - 1) * (d - 1)) // s + 1, p


def get_conv1d_out_shape(l, k, p, s=1, d=1):
    return get_conv1d_out_shape_and_pad(l, k, p, s, d)[0]


def get_conv2d_out_shape_and_pad(
    img_shape, ker_shape, padding="same", stride=(1, 1), dilation=(1, 1)
):
    if padding == "valid":
        padding = (0, 0)
    elif padding == "same":
        padding = ("s", "s")
    else:
        assert not isinstance(padding, str), padding
    return tuple(
        zip(
            *(
                get_conv1d_out_shape_and_pad(*x)
                for x in zip(img_shape, ker_shape, padding, stride, dilation)
            )
        )
    )


def get_conv2d_out_shape(
    img_shape, ker_shape, padding="same", stride=(1, 1), dilation=(1, 1)
):
    return get_conv2d_out_shape_and_pad(
        img_shape, ker_shape, padding, stride, dilation
    )[0]


def to_nchw(shape, layout):
    if layout == "nchw":
        n, c, h, w = shape
    elif layout == "nhwc":
        n, h, w, c = shape
    else:
        assert False
    return n, c, h, w


def from_nchw(shape, layout):
    n, c, h, w = shape
    if layout == "nchw":
        return n, c, h, w
    elif layout == "nhwc":
        return n, h, w, c
    else:
        assert False


def get_conv1d_inp_slice(out_slice: slice, out_len, inp_len, k, p, s=1, d=1):
    o_start, o_stop, _ = out_len.indices(out_slice)
    o_range = o_stop - o_start
    i_range = s * (o_range - 1) + k + (k - 1) * (d - 1)
    i_start = o_start
    i_stop = i_start + i_range
    i_start = clip(i_start - p, 0, inp_len)
    i_stop = clip(i_stop - p, 0, inp_len)
    inp_slice = slice(i_start, i_stop)
    return inp_slice

def clip(x, low, high):
    return min(high, max(low, x))
