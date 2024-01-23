import os

import torch
import inspect
import logging
import functools
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from awq.utils.utils import clear_memory
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.scale import apply_scale, apply_clip
from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
from awq.utils.module import append_str_prefix, get_op_name, get_named_linears, set_op_by_name


class AwqQuantizer:
    def __init__(self, awq_model, model, tokenizer, w_bit, group_size, version, visualize, vis_path,
                       calib_data, split, text_column, duo_scaling, modules_to_not_convert=None) -> None:
        self.quanting_layer_idx = None
        self.max_channel_idxes = []
        self.min_channel_idxes = []
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.version = version

        self.visualize = visualize
        self.vis_path = vis_path

        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.modules_to_not_convert = modules_to_not_convert if modules_to_not_convert is not None else []
        self.modules, self.module_kwargs, self.inps = self.init_quant()
    
    def pseudo_quantize_tensor(self, w: torch.Tensor, layer_name, get_scale_zp=False, scaled=False, ori_w=False):
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

        # scales [131072, 1]
        # w.shape [131072, 128]
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        # print("==> scales.shape: {}".format(scales.shape))
        # print("==> w.shape: {}".format(w.shape))
        # print("==> max_val.shape: {}".format(max_val.shape))
        # print("==> min_val.shape: {}".format(min_val.shape))
        # TODO: 这里也使用了clamp，有可能也有问题
        # 最小值里0之间的刻度值
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        # TODO: 这里也使用了clamp，有可能也有问题
        # 1. 计算torch.round(w / scales) + zeros，先shift到0开始，然后划分为最近临量化值（刻度值）
        # 2. shift回原位置，并将刻度值转回对应数值
        # TODO: 可视化scaled之前的（group设置成4096）
        w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros)

        if self.visualize and layer_name != "":
            self.print_channel_hist_plot(w.view(org_w_shape).cpu(), layer_name, self.max_channel_idxes[self.quanting_layer_idx],
                                         print_type="{}quant-wo-steped_{}_W-Dis_maxC"
                                         .format("ori-w_" if ori_w else "", "" if scaled else "wo-scaled_"))
            self.print_channel_hist_plot(w.view(org_w_shape).cpu(), layer_name, self.min_channel_idxes[self.quanting_layer_idx],
                                         print_type="{}quant-wo-steped_{}_W-Dis_minC"
                                         .format("ori-w_" if ori_w else "", "" if scaled else "wo-scaled_"))

        w = w * scales
        assert torch.isnan(w).sum() == 0

        w = w.reshape(org_w_shape)

        if get_scale_zp:
            return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
        else:
            return w
    
    def pseudo_dequantize_tensor(self, w: nn.Linear, scales: torch.Tensor, zeros: torch.Tensor):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // zeros.shape[-1]

        # get zeros and scales in correct shape
        zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        w = (w.weight.data - zeros) * scales

        return w
    
    def _exclude_layers_to_not_quantize(self, linear_layers):
        filtered_layers = {}
        for name, linear_layer in linear_layers.items():
            if not any(key in name for key in self.modules_to_not_convert):
                filtered_layers[name] = linear_layer
        return filtered_layers
    
    def quantize(self):
        # for block
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            self.quanting_layer_idx = i
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                self.modules[i] = self.modules[i].cuda()
                common_device = next(self.modules[i].parameters()).device

            # input dea
            self.inps = self.inps.to(common_device)

            # [STEP 1]: Get layer, extract linear modules, extract input features
            # 拿到模型中所有的线性层
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            # 从配置中减去self.modules_to_not_convert中包含的层
            named_linears = self._exclude_layers_to_not_quantize(named_linears)

            # 通过hook拿到每个block中线性层的输入feature

            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            # llama layers: List[Dict] => [attention input, attention out, linear 1, linear 2]
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )

            scales_list = [self._search_best_scale(self.modules[i], **layer) for layer in module_config]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(scales_list, get_op_name(self.model, self.modules[i]) + ".")

            # [STEP 3]: Compute and apply clipping list
            clip_list = self._search_best_clip(self.modules[i], named_linears, input_feat)
            apply_clip(self.modules[i], clip_list)
            clip_list = append_str_prefix(clip_list, get_op_name(self.model, self.modules[i]) + ".")

            # [STEP 4]: Quantize weights
            self._apply_quant(self.modules[i], named_linears)
            clear_memory()
    
    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.cuda().half()

            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data,
                layer_name="",
                get_scale_zp=True
            )

            if self.version == 'GEMM':
                scales = scales.t().contiguous()
                zeros = zeros.t().contiguous()
                q_linear_module = WQLinear_GEMM

            elif self.version  == 'GEMV':
                q_linear_module = WQLinear_GEMV
            
            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)
            clear_memory()

    @torch.no_grad()
    def _search_best_scale(self, module, layer_name, prev_op, layers: List[nn.Linear], inp: torch.Tensor, module2inspect=None, kwargs={}):
        print("==> _search_best_scale")
        # print("==> len(layers): {}".format(len(layers)))
        self.layer_name = layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]
        
        if "use_cache" in kwargs:
            kwargs.pop("use_cache")
        
        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute maximum of weight
        # [4096*3(QKV) = 12288, 4096]
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # print("==> weight.shape: {}".format(weight.shape))
        # [4096*(4096/128) = 391216, 128]
        weight = weight.view(-1, self.group_size) if self.group_size > 0 else weight
        # print("==> after grouped weight.shape: {}".format(weight.shape))
        # weight norm
        w_scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
        w_scale = w_scale.view(org_shape)
        # TODO:why mean?
        # [4096]
        w_max = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute maximum of x
        # [4096]
        x_max = inp.abs().view(-1, inp.shape[-1]).mean(0)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)

            fp16_output = module2inspect(inp, **module_kwargs)
            if isinstance(fp16_output, tuple):
                fp16_output = fp16_output[0]
        # print("==> fp16_output.shape: {}".format(fp16_output.shape))
        if self.visualize:
            self.print_channel_plot(fp16_output, layer_name, print_type="ori")

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_max, x_max, module2inspect,
            layer_name, layers, fp16_output, module_kwargs
        )

        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), best_scales)

    def print_channel_plot(self, fp16_output_, layer_name, print_type):
        # print("==> print_channel_plot")
        fp16_output = fp16_output_.view(-1, fp16_output_.shape[-1]).cpu()
        # print("==> fp16_output.shape: {}".format(fp16_output.shape))

        save_path = os.path.join(self.vis_path, "layer-{}".format(self.quanting_layer_idx), layer_name)
        # 保存图表为图片
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 计算每个通道的最大值和最小值
        max_values = torch.amax(fp16_output, dim=0)
        min_values = torch.amin(fp16_output, dim=0)
        # print("==> max_values.shape: {}".format(max_values.shape))

        max_max_value, max_max_channel_idx = torch.max(max_values, dim=0)
        min_min_value, min_min_channel_idx = torch.min(min_values, dim=0)
        # min_max_value, max_min_channel_idx = torch.min(max_values, dim=0)
        # max_min_value, min_max_channel_idx = torch.max(min_values, dim=0)
        self.max_channel_idxes.append(max_max_channel_idx)
        self.min_channel_idxes.append(min_min_channel_idx)

        # print channel hist plot
        self.print_channel_hist_plot(fp16_output, layer_name, self.max_channel_idxes[self.quanting_layer_idx],
                                     print_type="{}_C-Act-dis_maxC".format(print_type))
        self.print_channel_hist_plot(fp16_output, layer_name, self.min_channel_idxes[self.quanting_layer_idx],
                                     print_type="{}_C-Act-dis_minC".format(print_type))

        # 获取通道的数量（C）
        num_channels = fp16_output.shape[1]
        # 创建一个新的图表
        # fig, ax = plt.subplots(figsize=(20, 15))
        # font_size = 36  # 调整这个值来改变字体大小
        fig, ax = plt.subplots(figsize=(8, 6))
        font_size = 12  # 调整这个值来改变字体大小
        channel_ids = range(num_channels)
        max_bars = ax.bar(channel_ids, max_values, width=1, color='blue', label='Max Value')
        min_bars = ax.bar(channel_ids, min_values, width=1, color='orange', label='Min Value')
        # 标注最大值和最小值
        max_value_index = max_values.argmax().item()
        min_value_index = min_values.argmin().item()

        # 添加文本标签
        ax.text(max_value_index, max_values[max_value_index], f'Max Value: {max_values[max_value_index]:.2f}',
                ha='right', va='bottom', color='red', fontsize=font_size)
        ax.text(min_value_index, min_values[min_value_index], f'Min Value: {min_values[min_value_index]:.2f}',
                ha='right', va='bottom', color='red', fontsize=font_size)

        # 设置图表标题和轴标签
        ax.set_title('{} Max and Min Values Across Channels-layer{}-{}'
                     .format(print_type, self.quanting_layer_idx, layer_name), fontsize=font_size)
        ax.set_xlabel('Channel ID', fontsize=font_size)
        ax.set_ylabel('Values', fontsize=font_size)

        # 设置x轴范围为0到4096
        ax.set_xlim(0, num_channels)
        # 调整图表边缘紧贴
        # plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15)

        # 高亮离群值
        outlier_linewidth = 1
        for bar, value in zip(min_bars, min_values):
            if value <= min_values.max() * 10:
                bar.set_linewidth(outlier_linewidth)
                bar.set_edgecolor("orange")
        for bar, value in zip(max_bars, max_values):
            if value >= max_values.min() * 10:
                bar.set_linewidth(outlier_linewidth)
                bar.set_edgecolor("blue")

        # 调整y轴刻度标签字体大小
        ax.tick_params(axis='y', labelsize=font_size)
        ax.tick_params(axis='x', labelsize=font_size)

        plt.savefig(os.path.join(save_path, "{}_Channel-Activation.png".format(print_type)))

        # 显示图表
        # plt.show() # 假设fp16_output是你的 torch.tensor 向量
        plt.close()

    def print_channel_hist_plot(self, input, layer_name, channel_idx, print_type):
        # print("==> print_channel_hist_plot")
        values = input[:, channel_idx]

        save_path = os.path.join(self.vis_path, "layer-{}".format(self.quanting_layer_idx), layer_name)
        # 保存图表为图片
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = os.path.join(save_path, "{}_channel-{}.png"
                                 .format(print_type, channel_idx))

        fig, ax = plt.subplots(figsize=(8, 6))
        # 使用直方图表示激活值的分布情况
        ax.hist(values, bins=150, color='blue')

        # 标注最大值和最小值
        max_value = values.max()
        min_value = values.min()
        # 添加文本标签
        ax.text(max_value, 5, f'Max Value: {max_value:.2f}', ha='right', va='bottom', color='red', fontsize=10)
        ax.text(min_value, 5, f'Min Value: {min_value:.2f}', ha='left', va='bottom', color='red', fontsize=10)

        # 设置图表标题和轴标签
        if 'weight' in print_type or 'W' in print_type:
            ax.set_title('{} Distribution of Weight in Channel {}, Layer {}'
                         .format(print_type, channel_idx, self.quanting_layer_idx))
        else:
            ax.set_title('{} Distribution of Activations in Channel {}, Layer {}'
                         .format(print_type, channel_idx, self.quanting_layer_idx))
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

        # 保存图表为图片
        plt.savefig(save_path)

        plt.close()

    def _compute_best_scale(self, x, w_max, x_max, module2inspect, layer_name, linears2scale: List[nn.Linear],
                                  fp16_output, kwargs={}):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        print("==> _compute_best_scale")
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float('inf')

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}
        
        device = x.device
        # 其实是mean
        x_max = x_max.view(-1).to(device)
        w_max = w_max.view(-1).to(device)

        if self.visualize:
            max_idx = self.max_channel_idxes[self.quanting_layer_idx]
            min_idx = self.min_channel_idxes[self.quanting_layer_idx]

        # 使用网格搜索 1/1到20
        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid
        # if True:
        #     ratio = 0.6

            # NOTE: s^-1 * x is fused here, according to paper
            # 对应论文中的公式4，拿到 X * scale^-1, 此处用 X的平均来代替X
            # TODO: 这里用了clamp，有可能将过于小的权重给略过了
            if self.duo_scaling:
                scales = (x_max.pow(ratio) / w_max.pow(1-ratio)).clamp(min=1e-4)
            else:
                scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            # [4096]
            scales = scales / (scales.max() * scales.min()).sqrt()
            # [1, 4096]
            scales_view = scales.view(1, -1).to(device)

            # Q(W * s)
            # 拿到权重 W 量化并scale之后的权重
            for fc in linears2scale:
                # [4096, 4096]
                if self.visualize:
                    self.print_channel_hist_plot(fc.weight.cpu(), layer_name, max_idx,
                                                 print_type="ori_W-Dis_maxC")
                    self.print_channel_hist_plot(fc.weight.cpu(), layer_name, min_idx,
                                                 print_type="ori_W-Dis_minC")

                quant_wo_scaled_fc_weight_data = self.pseudo_quantize_tensor(fc.weight.data,
                                                                             layer_name=layer_name, ori_w=True)
                if self.visualize:
                    self.print_channel_hist_plot(quant_wo_scaled_fc_weight_data.cpu(), layer_name, max_idx,
                                                 print_type="ori-quanted_W-Dis_maxC")
                    self.print_channel_hist_plot(quant_wo_scaled_fc_weight_data.cpu(), layer_name, min_idx,
                                                 print_type="ori-quanted_W-Dis_minC")
                # [4096, 4096]
                fc.weight.mul_(scales_view)

                if self.visualize:
                    self.print_channel_hist_plot(fc.weight.cpu(), layer_name, max_idx,
                                                 print_type="scaled_W-Dis_maxC")
                    self.print_channel_hist_plot(fc.weight.cpu(), layer_name, min_idx,
                                                 print_type="scaled_W-Dis_minC")
                # RTN 最近邻量化

                fc.weight.data = self.pseudo_quantize_tensor(fc.weight.data, layer_name=layer_name, scaled=True) / scales_view
                if self.visualize:
                    self.print_channel_hist_plot(fc.weight.cpu(), layer_name, max_idx,
                                                 print_type="quanted_W-Dis_maxC")
                    self.print_channel_hist_plot(fc.weight.cpu(), layer_name, min_idx,
                                                 print_type="quanted_W-Dis_minC")

            # W * X
            int_w_output = module2inspect(x, **kwargs)
            if isinstance(int_w_output, tuple):
                int_w_output = int_w_output[0]
            # print("==> int_w_output.shape: {}".format(int_w_output.shape))

            # compute mean squared error (L2 norm)
            loss = (fp16_output - int_w_output).float().pow(2).mean().item() # NOTE: float prevents overflow

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
                self.print_channel_plot(int_w_output, layer_name, print_type="quanted")
                int_w_output_ = int_w_output.view(-1, int_w_output.shape[-1]).cpu()
                if self.visualize:
                    self.print_channel_hist_plot(int_w_output_, layer_name, max_idx,
                                                 print_type="quanted_C-Act-dis_maxC")
                    self.print_channel_hist_plot(int_w_output_, layer_name, min_idx,
                                                 print_type="quanted_C-Act-dis_minC")
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales
        print("==> best_ratio: {}".format(best_ratio))
        return best_scales.detach().cpu()

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].cuda()
            print("==> name: {}".format(name))
            max_val = self._compute_best_clip(named_linears[name].weight, input_feat[name])
            clip_list.append((name, max_val))

            named_linears[name].cpu()
        
        return clip_list

    @torch.no_grad()
    def _compute_best_clip(self, w: torch.Tensor, input_feat: torch.Tensor, n_grid=20, max_shrink=0.5, n_sample_token=512):
        assert w.dim() == 2
        org_w_shape = w.shape
        # org_w_shape: torch.Size([4096, 11008])
        # input_feat.shape: torch.Size([65, 512, 11008])
        # print("==> _compute_best_clip org_w_shape: {}".format(org_w_shape))
        # print("==> _compute_best_clip input_feat.shape: {}".format(input_feat.shape))
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else w.shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
        w = w.reshape(w.shape[0], 1, -1, group_size)

        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = - max_val
                cur_w = torch.clamp(w, min_val, max_val)
                # print("==> cur_w.shape: {}".format(cur_w.shape))
                q_w = self.pseudo_quantize_tensor(cur_w, layer_name="")
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    def init_quant(self, n_samples=128, seqlen=512):
        print("==> init_quant")
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data, tokenizer=self.tokenizer, n_samples=n_samples, block_size=seqlen,
            split=self.split, text_column=self.text_column
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        modules[0] = modules[0].cuda()
        self.awq_model.move_embed(self.model, "cuda")
        
        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        
        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        modules[0] = modules[0].module  # restore
        print("==> len(inps): {}".format(len(inps)))
        print("==> inps[0].shape: {}".format(inps[0].shape))
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")
        
        clear_memory()
        
        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to("cuda")

        return modules, layer_kwargs, inps
    
    def _get_input_feat(self, layer, named_linears):
        print("==> _get_input_feat")
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                feat_dict=input_feat)))
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # [bs, sequence_len, hidden state] -> [65, 512, 4096]
        # "hidden_size": 4096,
        # "initializer_range": 0.02,
        # "intermediate_size": 11008,
        print("==> self.inps.shape: {}".format(self.inps.shape))
        # get output as next layer's input
        
        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = layer(self.inps, **module_kwargs)[0]
        print("==> after forward self.inps.shape: {}".format(self.inps.shape))

        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        # ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        # 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        # print("==> input_feat.keys(): {}".format(input_feat.keys()))
        
        return input_feat


    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers. 

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in  inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs