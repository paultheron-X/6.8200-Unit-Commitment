import os
import torch
import argparse

import coloredlogs

from config import return_config

torch.cuda.empty_cache()
torch.set_printoptions(precision=10)
torch.autograd.set_detect_anomaly(True)

def _parse_args():
    parser = argparse.ArgumentParser(description="Run VAE for FaceGeneration")
    parser.add_argument(
        "--config",
        "-c",
        default="tpl",
        choices=["tpl"],
        help="path to config file",
    )
    parser.add_argument(
        "--verbose", "-v", help="Sets the lebel of verbose", action="store_true"
    )    
    return parser.parse_args()


def main(args):
    pass

if __name__ == '__main__':
    args = _parse_args()
    level = "INFO" if not args.verbose else "DEBUG"
    config = dict(
        fmt="[[{relativeCreated:7,.0f}ms]] {levelname} [{module}] {message}",
        style="{",
        level=level,
    )
    coloredlogs.DEFAULT_LEVEL_STYLES["debug"] = {"color": 201}
    coloredlogs.DEFAULT_LEVEL_STYLES["warning"] = {
        "color": "red",
        "style": "bright",
        "bold": True,
        "italic": True,
    }
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = {
        "color": "blue",
        "style": "bright",
        "bold": True,
    }
    coloredlogs.DEFAULT_FIELD_STYLES["relativeCreated"] = {"color": 10, "style": "bright"}

    coloredlogs.install(**config)
    main(args)