"""Load inherited YAML files and adapt them to the processing runtime."""

import copy
import os
from pathlib import Path

import yaml


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(config_file, parents=None):
    config_file = Path(config_file).resolve()
    parents = [] if parents is None else parents
    if config_file in parents:
        chain = " -> ".join(str(path) for path in parents + [config_file])
        raise ValueError(f"cyclic configuration inheritance: {chain}")

    with config_file.open("r") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"configuration root must be a mapping: {config_file}")

    inherited = config.pop("extends", [])
    inherited = [inherited] if isinstance(inherited, str) else inherited
    merged = {}
    for parent in inherited:
        merged = _deep_merge(
            merged,
            _load_yaml(config_file.parent / parent, parents + [config_file]),
        )
    return _deep_merge(merged, config)


def _normalize(config):
    options = config["options"]
    if isinstance(options.get("sensor"), str):
        options["sensor"] = [options["sensor"]]

    # Promote the old monolithic structure when loading legacy configuration files.
    legacy_step = options.pop("proc_step", None)
    legacy_stage_options = options.pop("proc_step_options", {})
    config.setdefault("stage", legacy_step)
    for name, settings in legacy_stage_options.items():
        config.setdefault(name, settings)

    stage = config["stage"]

    if stage == "stacking":
        stacking = config["stacking"]
        parallel = stacking["parallel"]
        stacking["multiproc"] = parallel["enabled"]
        stacking["num_cpus"] = parallel["workers"]

        climatology = stacking["climatology"]
        config.setdefault("auxiliary", {})["sit_clim"] = climatology["sit"].get("path")
        config["auxiliary"]["tfb_clim"] = climatology["tfb"].get("path")

        thermo = stacking["thermo_change"]
        options["t2m_product"] = thermo["air_temperature"]["product"]
        heat_flux = thermo["ocean_heat_flux"]
        if heat_flux["source"] == "constant":
            thermo["oce_heat_flux"] = heat_flux["value"]
        else:
            options["ohf_product"] = heat_flux["product"]
            thermo["oce_heat_flux"] = heat_flux["product"]

    elif stage == "gridding":
        gridding = config["gridding"]
        parallel = gridding["parallel"]
        gridding["multiproc"] = parallel["enabled"]
        gridding["num_cpus"] = parallel["workers"]
        gridding["weighting"]["is_weight"] = gridding["weighting"]["enabled"]

    elif stage == "visualization":
        visualization = config["visualization"]
        visualization["make_gif"] = visualization["gif"]["enabled"]

    return config


def _resolve_paths(config):
    sensors = config["options"]["sensor"]
    sensor_name = "_".join(sensors)
    version = config["version"]

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            return value.format(sensor=sensor_name, version=version)
        return value

    config = expand(config)
    base_dir = config["base_dir"]

    def absolute(value):
        if isinstance(value, dict):
            return {key: absolute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [absolute(item) for item in value]
        if isinstance(value, str) and value and not os.path.isabs(value):
            return os.path.normpath(os.path.join(base_dir, value))
        return value

    for section in ("input_dir", "output_dir", "auxiliary"):
        config[section] = absolute(config[section])
    config["logging"] = absolute(config["logging"])
    return config


def load_config(config_file):
    """Load, merge, normalize, and resolve a stage configuration."""
    return _resolve_paths(_normalize(_load_yaml(config_file)))
