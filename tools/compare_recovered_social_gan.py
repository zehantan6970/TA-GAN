"""Compare restored Social-GAN parameters with retained Python 3.6 bytecode.

Run this script with the legacy ``ta-gan`` environment because CPython 3.8
cannot import a CPython 3.6 ``.pyc``.  It compares seeded initialization,
state-dict names, order, shapes, and values; no forward pass is executed.
"""

from __future__ import print_function

import argparse
import importlib.machinery
import importlib.util
import os
import sys

import torch


# Import restored source without replacing the historical .pyc being audited.
sys.dont_write_bytecode = True


def load_sourceless_module(path):
    loader = importlib.machinery.SourcelessFileLoader(
        "historic_social_gan_model", path
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def build_generator(module):
    return module.TrajectoryGenerator(
        obs_len=4,
        pred_len=3,
        embedding_dim=8,
        encoder_h_dim=8,
        decoder_h_dim=12,
        mlp_dim=16,
        noise_dim=(4,),
        pooling_type="pool_net",
        bottleneck_dim=8,
        batch_norm=False,
    )


def main(pyc_path, package_root):
    if sys.version_info[:2] != (3, 6):
        raise RuntimeError("Run with Python 3.6; found {}".format(sys.version))

    sys.path.insert(0, package_root)
    from sgan import models as restored_module

    historic_module = load_sourceless_module(pyc_path)
    torch.manual_seed(2026)
    historic = build_generator(historic_module)
    torch.manual_seed(2026)
    restored = build_generator(restored_module)

    historic_state = historic.state_dict()
    restored_state = restored.state_dict()
    if list(historic_state.keys()) != list(restored_state.keys()):
        raise AssertionError("state-dict parameter names or order differ")

    for name in historic_state:
        historic_value = historic_state[name]
        restored_value = restored_state[name]
        if historic_value.shape != restored_value.shape:
            raise AssertionError("shape differs for {}".format(name))
        if not torch.equal(historic_value, restored_value):
            raise AssertionError(
                "seeded initialization differs for {}".format(name)
            )

    print(
        "MATCH parameters={} tensors={}".format(
            sum(value.numel() for value in historic_state.values()),
            len(historic_state),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyc", required=True)
    parser.add_argument("--package-root", required=True)
    args = parser.parse_args()
    main(os.path.abspath(args.pyc), os.path.abspath(args.package_root))
