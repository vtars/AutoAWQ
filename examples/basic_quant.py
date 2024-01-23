import argparse
from time import strftime, localtime

import torch

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def main(args):
    quant_config = {"visualize": args.visualize, "zero_point": args.zero_point, "vis_path": args.vis_path,
                    "q_group_size": args.q_group_size, "w_bit": args.w_bit, "version": args.version}

    # Load model
    # NOTE: pass safetensors=True to load safetensors
    model = AutoAWQForCausalLM.from_pretrained(args.model_path, **{"low_cpu_mem_usage": False})
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(args.quant_path)
    tokenizer.save_pretrained(args.quant_path)

    print(f'Model is quantized and saved at "{args.quant_path}"')


if __name__ == "__main__":
    print(torch.nn.Parameter(torch.tensor(6.0)).shape)
    exit(1)
    time_step = strftime('%Y%m%d-%H%M', localtime())
    print("==> time(): {}".format(time_step))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/home/data/dixiang/huggingface/llama-7b-hf",
                        help="path to the model")
    parser.add_argument("--quant_path", type=str,
                        default="/home/dixiang/hsx/projects/AutoAWQ/output/test-vis-llama-7b-hf_{}".format(time_step),
                        help="path to save quanted weights")
    parser.add_argument("--visualize", default=False, action="store_true", help="print visualization plot")
    parser.add_argument("--vis_path", type=str,
                        default="/home/dixiang/hsx/projects/AutoAWQ/visualization/llama-7b-test_{}".format(time_step),
                        help="path to save vis polt")
    parser.add_argument("--zero_point", default=True, action="store_true", help="shift to zero point")
    parser.add_argument("--q_group_size", type=int, default=128, help="quant group size")
    parser.add_argument("--w_bit", type=int, default=4, help="Bits of quanted weights")
    parser.add_argument("--version", type=str, default="GEMM", choices=["GEMM", "GEMV"], help="quent version")
    args = parser.parse_args()

    main(args)
