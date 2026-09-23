"""Prepare local paths for the shipped geometry v2 artifacts without training."""
from pathlib import Path
import os
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "data/experiments/ml_weather_geometry_v1/runtime_data"
    for site in ("turbine_1", "turbine_2"):
        if not list((root / "models/geometry_v2/active" / site).glob("ml-*/model.cbm")):
            raise FileNotFoundError(f"Saved geometry v2 models missing for {site}")
    if not (runtime / "ml/final_models.json").is_file():
        raise FileNotFoundError(runtime / "ml/final_models.json")
    for name in ("weather", "scada"):
        source, destination = root / "data" / name, runtime / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"Unexpected existing data link: {destination}")
            # Replace only an equivalent absolute link from the original machine.
            if destination.readlink().is_absolute():
                destination.unlink()
            else:
                continue
        elif destination.exists():
            if not destination.is_dir():
                raise ValueError(f"Expected a directory: {destination}")
            continue
        try:
            destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)
        except OSError:
            # Windows without symlink privileges: retain the existing cache format.
            shutil.copytree(source, destination)
    config = root / "config.local.json"
    if not config.exists():
        shutil.copyfile(root / "config.example.json", config)
    print(f"Geometry v2 runtime ready: {runtime}")


if __name__ == "__main__":
    main()
