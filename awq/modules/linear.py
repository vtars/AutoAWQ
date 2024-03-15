import math
import torch
import torch.nn as nn
import awq_inference_engine  # with CUDA kernels


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


class WQLinear_GEMM(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, a_quant=False):
        super().__init__()
        print("==> __init__ WQLinear_GEMM")

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.a_quant = a_quant

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        # [in_feat_dim, pack_num] => [4096, 512]
        self.register_buffer('qweight',
                             torch.zeros((in_features, out_features // (32 // self.w_bit)), dtype=torch.int32,
                                         device=dev))

        # [group_num, pack_num] => [32, 4096]
        self.register_buffer('qzeros', torch.zeros((in_features // self.group_size, out_features // (32 // self.w_bit)),
                                                   dtype=torch.int32, device=dev))
        # [group_num, out_feat_dim] => [32, 4096]
        self.register_buffer('scales', torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16,
                                                   device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, a_quant=False, scales=None, zeros=None):
        print("==> from_linear")
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None,
                         linear.weight.device, a_quant)
        if init_only:  # just prepare for loading sd
            return awq_linear

        # print("==> qweight: {}".format(awq_linear.qweight.shape))
        # print("==> qzeros: {}".format(awq_linear.qzeros.shape))
        # print("==> scales: {}".format(awq_linear.scales.shape))
        # exit(-1)

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        intweight = []
        # for each token
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round(
                (linear.weight.data[:, idx] + scale_zeros[idx // group_size]) / awq_linear.scales[
                    idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32,
                              device=intweight.device)

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32,
                             device=zeros.device)

        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros

        return awq_linear

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        # RTN

        # w: grouped weights [131072, 128]
        # print("==> pseudo_quantize_tensor")
        org_w_shape = w.shape
        if self.group_size > 0:
            # print("==> org_w_shape: {}, layer_name: {}".format(org_w_shape, layer_name))
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        elif self.group_size == 0:
            w = w.reshape(-1, w.shape[-1])
        assert w.dim() == 2

        # zero point quantization
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** self.w_bit - 1
        min_int = 0
        # 这里计算量化，这个scale跟AWQ里面的scale不一样，这里指的是刻度长度
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        # 最小值离0之间的刻度值
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        # TODO: 这里也使用了clamp，有可能也有问题
        # 1. 计算torch.round(w / scales) + zeros，先shift到0开始，然后划分为最近临量化值（刻度值）
        # 2. shift回原位置，并将刻度值转回对应数值
        w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros)

        w = w * scales
        assert torch.isnan(w).sum() == 0

        w = w.reshape(org_w_shape)

        return w

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        if self.a_quant:
            x = self.pseudo_quantize_tensor(x)
        else:
            input_dtype = x.dtype
            if input_dtype != torch.float16:
                x = x.half()

        out = awq_inference_engine.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros,
                                                     8)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )


class WQLinear_GEMV(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, a_quant=False):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = (32 // self.w_bit)

        self.register_buffer('qweight',
                             torch.zeros((out_features, in_features // pack_num), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size)),
                                                   dtype=torch.int32, device=dev))
        self.register_buffer('scales',
                             torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size) * pack_num),
                                         dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, a_quant=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None,
                         linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.w_bit
        qscales = torch.zeros(
            (scales.shape[0], calculate_zeros_width(linear.in_features, group_size) * pack_num),
            dtype=torch.float16,
            device=scales.device
        )
        qscales[:, :scales.shape[1]] = scales
        awq_linear.scales = qscales
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round(
                (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size]) / awq_linear.scales[:,
                                                                                   idx // group_size]).to(torch.int)[:,
                             None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32,
                              device=intweight.device)

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros(
            (zeros.shape[0], calculate_zeros_width(linear.in_features, group_size)),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range((zeros.shape[1] + pack_num - 1) // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                if col * pack_num + order_map[i] >= zeros.shape[1]:
                    continue
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        inputs = x.reshape(-1, x.shape[-1])

        input_dtype = inputs.dtype
        if input_dtype != torch.float16:
            inputs = inputs.half()

        if inputs.shape[0] > 8:
            out = awq_inference_engine.gemmv2_forward_cuda(inputs, self.qweight, self.scales, self.qzeros,
                                                           self.group_size, self.split_k_iters)
        else:
            out = awq_inference_engine.gemv_forward_cuda(inputs, self.qweight, self.scales, self.qzeros,
                                                         self.group_size)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )
