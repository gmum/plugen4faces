from collections import abc
import os

import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

from . import conv2d_gradfix

module_path = os.path.dirname(__file__)
if torch.cuda.is_available():
    upfirdn2d_op = load(
        "upfirdn2d",
        sources=[
            os.path.join(module_path, "upfirdn2d.cpp"),
            os.path.join(module_path, "upfirdn2d_kernel.cu"),
        ],
    )
else:

    class upfirdn2d_op:
        @staticmethod
        def upfirdn2d(x, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
            """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops."""
            # Validate arguments.
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            assert f.dtype == torch.float32 and not f.requires_grad
            batch_size, num_channels, in_height, in_width = x.shape

            # Upsample by inserting zeros.
            x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
            x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
            x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

            # Pad or crop.
            x = torch.nn.functional.pad(
                x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)]
            )
            x = x[
                :,
                :,
                max(-pady0, 0) : x.shape[2] - max(-pady1, 0),
                max(-padx0, 0) : x.shape[3] - max(-padx1, 0),
            ]

            # Setup filter.
            f = f * (gain ** (f.ndim / 2))
            f = f.to(x.dtype)
            if not flip_filter:
                f = f.flip(list(range(f.ndim)))

            # Convolve with the filter.
            f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
            if f.ndim == 4:
                x = conv2d_gradfix.conv2d(input=x, weight=f, groups=num_channels)
            else:
                x = conv2d_gradfix.conv2d(
                    input=x, weight=f.unsqueeze(2), groups=num_channels
                )
                x = conv2d_gradfix.conv2d(
                    input=x, weight=f.unsqueeze(3), groups=num_channels
                )

            # Downsample by throwing away pixels.
            x = x[:, :, ::downy, ::downx]
            return x


class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        (kernel,) = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = UpFirDn2dBackward.apply(
                grad_output,
                kernel,
                grad_kernel,
                ctx.up,
                ctx.down,
                ctx.pad,
                ctx.g_pad,
                ctx.in_size,
                ctx.out_size,
            )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    if input.device.type == "cpu":
        out = upfirdn2d_native(input, kernel, *up, *down, *pad)

    else:
        out = UpFirDn2d.apply(input, kernel, up, down, pad)

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    return out.view(-1, channel, out_h, out_w)
