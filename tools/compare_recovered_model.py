"""Compare recovered source parameters with the retained Python 3.6 module.

This script must run under Python 3.6 because it imports the historical
``models_transformer.cpython-36.pyc`` directly. It does not execute a forward
pass; the historical forward method hard-codes CUDA noise allocation.
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
    loader = importlib.machinery.SourcelessFileLoader("historic_model", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def build_generator(module):
    return module.Trajectory_Generator(
        obs_len=8,
        embedding_dim=16,
        encoder_input_dim=16,
        encoder_output_dim=16,
        encoder_mlp_dim=16,
        encoder_num_head=2,
        drop_rate=0,
        rel_traj_dim=16,
        noise_dim=4,
        merge_mlp_dim=16,
    )


def main(pyc_path, package_root):
    if sys.version_info[:2] != (3, 6):
        raise RuntimeError("Run with Python 3.6; found {}".format(sys.version))

    sys.path.insert(0, package_root)
    from sgan import models_transformer as recovered_model

    historic_model = load_sourceless_module(pyc_path)

    torch.manual_seed(2026)
    historic = build_generator(historic_model)
    torch.manual_seed(2026)
    recovered = build_generator(recovered_model)

    historic_state = historic.state_dict()
    recovered_state = recovered.state_dict()
    if list(historic_state.keys()) != list(recovered_state.keys()):
        raise AssertionError("state-dict parameter names or order differ")

    for name in historic_state:
        historic_value = historic_state[name]
        recovered_value = recovered_state[name]
        if historic_value.shape != recovered_value.shape:
            raise AssertionError("shape differs for {}".format(name))
        if not torch.equal(historic_value, recovered_value):
            raise AssertionError("seeded initialization differs for {}".format(name))

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

