from types import SimpleNamespace

import pytest
import torch

from espnet2.tasks.gan_codec import GANCodecTask


def test_add_arguments():
    GANCodecTask.get_parser()


def test_add_arguments_help():
    parser = GANCodecTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        GANCodecTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        GANCodecTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        GANCodecTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        GANCodecTask.print_config(f)
    parser = GANCodecTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


class _OptimizerTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.embedding = torch.nn.Embedding(8, 4)


class _OptimizerTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.codec = torch.nn.Module()
        self.codec.generator = _OptimizerTestModule()
        self.codec.discriminator = _OptimizerTestModule()


def _optimizer_args(exclude_weight_decay):
    optimizer_conf = {"lr": 1.0e-3, "weight_decay": 0.01}
    return SimpleNamespace(
        optim="adamw",
        optim_conf=optimizer_conf,
        optim2="adamw",
        optim2_conf=optimizer_conf,
        exclude_weight_decay=exclude_weight_decay,
        exclude_weight_decay_conf={},
        sharded_ddp=False,
    )


def _weight_decay_by_parameter(optimizer):
    return {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_optimizer_weight_decay_exclusions_are_opt_in():
    model = _OptimizerTestModel()
    optimizers = GANCodecTask.build_optimizers(_optimizer_args(False), model)

    for module, optimizer in zip(
        (model.codec.generator, model.codec.discriminator), optimizers
    ):
        decay = _weight_decay_by_parameter(optimizer)
        assert len(decay) == len(list(module.parameters()))
        assert set(decay.values()) == {0.01}
        assert {group["lr"] for group in optimizer.param_groups} == {1.0e-3}


def test_optimizer_excludes_bias_norm_and_embedding():
    model = _OptimizerTestModel()
    optimizers = GANCodecTask.build_optimizers(_optimizer_args(True), model)

    for module, optimizer in zip(
        (model.codec.generator, model.codec.discriminator), optimizers
    ):
        decay = _weight_decay_by_parameter(optimizer)
        assert len(decay) == len(list(module.parameters()))
        assert decay[id(module.linear.weight)] == 0.01
        assert decay[id(module.linear.bias)] == 0.0
        assert decay[id(module.norm.weight)] == 0.0
        assert decay[id(module.norm.bias)] == 0.0
        assert decay[id(module.embedding.weight)] == 0.0
        assert {group["lr"] for group in optimizer.param_groups} == {1.0e-3}
