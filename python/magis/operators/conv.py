from .base import Operator, ComputeOp
from .. import utils
from typing import List, Tuple


class Conv2dOp(ComputeOp):
    def __init__(
        self,
        img: Operator,
        wgt: Operator,
        layout=None,
        padding=0,
        stride=1,
        dilation=1,
    ) -> None:
        layout = layout or Operator.DEFAULT_IMAGE_LAYOUT
        img_shape = img.out_shape
        wgt_shape = wgt.out_shape
        padding = padding if isinstance(padding, str) else utils.to_nd_tuple(padding)
        stride = utils.to_nd_tuple(stride)
        dilation = utils.to_nd_tuple(dilation)

        n, c, h, w = utils.to_nchw(img_shape, layout)
        k, c1, r, s = utils.to_nchw(wgt_shape, layout)

        assert c == c1
        out_shape, padding = utils.get_conv2d_out_shape_and_pad(
            (h, w), (r, s), padding, stride, dilation
        )
        out_shape = utils.from_nchw((n, k, *out_shape), layout)

        super().__init__(
            "conv2d",
            out_shape,
            [img, wgt],
            layout=layout,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )

        assert layout == "nchw"
        # we currently do not consider connections of height & width dims
        self._dim_links = [[(0, 0), (1, -1)], [(0, 1), (1, -1)]]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        padding = self.attrs["padding"]
        stride = self.attrs["stride"]
        dilation = self.attrs["dilation"]
        layout = self.attrs["layout"]
        assert layout == "nchw"
        img, wgt = inps
        ig = Conv2dBwdInpOp(
            img.out_shape, out_grad, wgt, layout, padding, stride, dilation
        )
        wg = Conv2dBwdWgtOp(
            wgt.out_shape, img, out_grad, layout, padding, stride, dilation
        )
        return [ig, wg], [(ig, [out_grad, wgt]), (wg, [img, out_grad])]


# ugly
class Conv2dBwdInpOp(ComputeOp):
    REUSE_PROF_CODE = False

    def __init__(
        self,
        img_shape: List[int],
        out_grad: Operator,
        wgt: Operator,
        layout,
        padding,
        stride,
        dilation,
    ) -> None:
        super().__init__(
            "conv2d_bwd_inp",
            img_shape,
            [out_grad, wgt],
            layout=layout,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )

        assert layout == "nchw"
        self._dim_links = [[(0, 0), (1, -1)], [(0, -1), (1, 1)]]


# ugly
class Conv2dBwdWgtOp(ComputeOp):
    REUSE_PROF_CODE = False

    def __init__(
        self,
        wgt_shape: List[int],
        img: Operator,
        out_grad: Operator,
        layout,
        padding,
        stride,
        dilation,
    ) -> None:
        super().__init__(
            "conv2d_bwd_wgt",
            wgt_shape,
            [img, out_grad],
            layout=layout,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )

        assert layout == "nchw"
        self._dim_links = [[(0, -1), (1, 1)], [(0, -1), (1, 0)]]


class Pool2dOp(ComputeOp):
    def __init__(
        self,
        tag: str,
        img: Operator,
        kernel_size,
        layout=None,
        padding=0,
        stride=1,
    ) -> None:
        layout = layout or Operator.DEFAULT_IMAGE_LAYOUT
        img_shape = img.out_shape
        kernel_size = utils.to_nd_tuple(kernel_size)
        padding = padding if isinstance(padding, str) else utils.to_nd_tuple(padding)
        stride = utils.to_nd_tuple(stride)

        n, c, h, w = utils.to_nchw(img_shape, layout)

        out_shape, padding = utils.get_conv2d_out_shape_and_pad(
            (h, w), kernel_size, padding, stride
        )
        out_shape = utils.from_nchw((n, c, *out_shape), layout)

        super().__init__(
            f"pool2d.{tag}",
            out_shape,
            [img],
            kernel_size=kernel_size,
            layout=layout,
            padding=padding,
            stride=stride,
        )

        assert layout == "nchw"
        self._dim_links = [[(0, 0), (1, 1)]]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        padding = self.attrs["padding"]
        stride = self.attrs["stride"]
        layout = self.attrs["layout"]
        kernel_size = self.attrs["kernel_size"]
        assert layout == "nchw"
        (img,) = inps
        tag = self.tag.split(".")[-1]
        ig = Pool2dBwdOp(
            tag,
            out_grad,
            img,
            kernel_size=kernel_size,
            layout=layout,
            padding=padding,
            stride=stride,
        )
        return [ig], [(ig, [out_grad, img])]


# ugly
class Pool2dBwdOp(ComputeOp):
    def __init__(
        self,
        tag: str,
        out_grad: Operator,
        img: Operator,
        kernel_size,
        layout,
        padding,
        stride,
    ) -> None:
        assert tag == "avg"
        super().__init__(
            f"pool2d_bwd.{tag}",
            img.out_shape,
            [out_grad, img],
            kernel_size=kernel_size,
            layout=layout,
            padding=padding,
            stride=stride,
        )

        assert layout == "nchw"
        self._dim_links = [[(0, 0), (1, 1)]]


class InterpolateOp(ComputeOp):
    def __init__(
        self, inp: Operator, size=None, scale=None, mode="nearest", layout=None
    ) -> None:
        layout = layout or Operator.DEFAULT_IMAGE_LAYOUT
        assert layout == "nchw"
        n, c, h, w = utils.to_nchw(inp.out_shape, layout)
        assert (size and not scale) or (not size and scale)
        if size:
            size = utils.to_nd_tuple(size)
            out_shape = (n, c, *size)
        else:
            scale = utils.to_nd_tuple(scale)
            out_shape = (n, c, utils.floor(h * scale[0]), utils.floor(w * scale[1]))
        out_shape = utils.from_nchw(out_shape, layout)

        super().__init__(
            "interpolate",
            out_shape,
            [inp],
            size=size,
            scale=scale,
            mode=mode,
            layout=layout,
        )

        assert layout == "nchw"
        self._dim_links = [[(0, 0), (1, 1)]]

    def backward(
        self, out_grad: "Operator", inps: List["Operator"]
    ) -> Tuple[List["Operator"], List[Tuple["Operator", List["Operator"]]]]:
        mode = self.attrs["mode"]
        layout = self.attrs["layout"]
        assert layout == "nchw"
        assert mode in ("nearest", "bilinear", "bicubic")
        (img,) = inps
        ig = InterpolateBwdOp(out_grad, img.out_shape, layout, mode)
        return [ig], [(ig, [out_grad])]


# ugly
class InterpolateBwdOp(ComputeOp):
    REUSE_PROF_CODE = False

    def __init__(self, out_grad: Operator, inp_shape, layout, mode) -> None:
        _, _, h, w = utils.to_nchw(out_grad.out_shape, layout)
        super().__init__(
            "interpolate_bwd",
            inp_shape,
            [out_grad],
            out_size=(h, w),
            layout=layout,
            mode=mode,
        )

        assert layout == "nchw"
        self._dim_links = [[(0, 0), (1, 1)]]
