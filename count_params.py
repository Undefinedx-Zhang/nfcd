import argparse
import json
from types import SimpleNamespace
import torch

import models
from models.nf import load_decoder_arch


def count_module_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def build_model(cfg, num_classes):
    backbone = cfg["model"]["backbone"]
    factory = {
        "ResNet50": models.NF_ResNet50_CD,
        "ResNet101": models.NF_ResNet101_CD,
        "HRNet": models.NF_HRNet_CD,
        "NF": models.NF_ResNet50_CD,  # legacy alias
    }
    if backbone not in factory:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return factory[backbone](num_classes=num_classes, config=cfg, testing=True)


def build_nf_decoders(cfg):
    nf_conf = SimpleNamespace(**cfg["nf_trainer"])
    return [load_decoder_arch(nf_conf, dim) for dim in nf_conf.pool_dims]


def main():
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument(
        "--config",
        default="configs/config_WHU.json",
        help="Path to config JSON",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes for the change detector head",
    )
    parser.add_argument(
        "--include-nf",
        action="store_true",
        help="Also count NF decoder parameters (stage2)",
    )
    args = parser.parse_args()

    cfg = json.load(open(args.config))

    model = build_model(cfg, args.num_classes)
    base_total, base_trainable = count_module_params(model)

    nf_total = 0
    if args.include_nf:
        nf_decoders = build_nf_decoders(cfg)
        nf_total = sum(p.numel() for d in nf_decoders for p in d.parameters())

    print(f"Backbone: {cfg['model']['backbone']}")
    print(
        f"Base model params: total={base_total/1e6:.2f}M "
        f"trainable={base_trainable/1e6:.2f}M"
    )
    if args.include_nf:
        print(f"NF decoders params: {nf_total/1e6:.2f}M")
        print(f"All params (base + NF decoders): {(base_total + nf_total)/1e6:.2f}M")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
