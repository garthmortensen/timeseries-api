
import pytest
import yaml
from utilities.configurator import (
    read_config_from_fs,
    Config
)

sample_config = {
    "metadata_version": 1.0,
    "metadata_environment": "dev",
    "source_actual_or_synthetic_data": "synthetic",
    "data_start_date": "2023-01-01",
    "data_end_date": "2023-12-31",
    "symbols": ["GME", "BYND"],
    "synthetic_anchor_prices": [150.5, 700.0],
    "synthetic_random_seed": 42,
    "data_processor_missing_values_strategy": "drop",
    "data_processor_scaling_method": "standardize",
}

@pytest.fixture
def config_file(tmp_path):
    config_path = tmp_path / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path

def test_read_config_from_fs(config_file):
    config = read_config_from_fs(config_file)
    assert config["data_start_date"] == "2023-01-01"
    assert config["data_end_date"] == "2023-12-31"
    assert config["synthetic_anchor_prices"] == [150.5, 700.0]

def test_config_model():
    config = Config(**sample_config)
    assert config.data_start_date == "2023-01-01"
    assert config.data_processor_missing_values_strategy == "drop"
    assert config.synthetic_anchor_prices == [150.5, 700.0]
    assert config.source_actual_or_synthetic_data == "synthetic"
