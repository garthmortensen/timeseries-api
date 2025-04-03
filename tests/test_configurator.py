# tests/test_configurator.py

import pytest
import yaml
from utilities.configurator import (
    read_config_from_fs,
    Config
)

sample_config = {
    "metadata_version": 1.0,
    "metadata_environment": "dev",
    "data_generator_enabled": True,
    "data_generator_start_date": "2023-01-01",
    "data_generator_end_date": "2023-12-31",
    "data_generator_anchor_prices_GME": 150.5,
    "data_generator_anchor_prices_BYND": 700.0,
    "data_processor_missing_values_strategy": "drop",
    "data_processor_scaling_method": "standardize",
}


@pytest.fixture
# fixture that returns a path to a temp file
def config_file(tmp_path):
    config_path = tmp_path / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


def test_read_config_from_fs(config_file):
    config = read_config_from_fs(config_file)
    assert config["data_generator_start_date"] == "2023-01-01"
    assert config["data_generator_end_date"] == "2023-12-31"
    assert config["data_generator_anchor_prices_GME"] == 150.5


def test_config_model():
    config = Config(**sample_config)
    assert config.data_generator_start_date == "2023-01-01"
    assert config.data_processor_missing_values_strategy == "drop"
    assert config.data_generator_anchor_prices_GME == 150.5
