#!/usr/bin/env python
"""
run_pipeline.py
~~~~~~~~~~~~~~~
CLI entry point for the cross-lingual voice cloning pipeline.

Usage
-----
python run_pipeline.py --config config/config.yaml

Optional flags
--------------
--config      Path to YAML config (default: config/config.yaml)
--models      Space-separated list of model names to run (overrides config)
--langs       Space-separated list of language codes to generate (overrides config)
--max-samples Maximum number of dataset samples to process
--dry-run     Print what would be generated without actually running models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is in sys.path when invoked directly
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models import build_model
from pipeline.dataset_processor import DatasetProcessor
from pipeline.generation_pipeline import GenerationPipeline
from utils.config import enabled_models, load_config
from utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-Lingual Voice Cloning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="Override enabled models (e.g. --models CosyVoice3 VoxCPM).",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        metavar="LANG",
        default=None,
        help="Override target languages (e.g. --langs fr zh ar).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap number of dataset samples to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + dataset but do not load models or generate audio.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load configuration ────────────────────────────────────────────── #
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file"),
    )

    logger.info("=" * 60)
    logger.info("Cross-Lingual Voice Cloning Pipeline")
    logger.info("Config: %s", args.config)
    logger.info("=" * 60)

    # ── Resolve settings (CLI overrides config) ───────────────────────── #
    target_langs: list[str] = args.langs or cfg["target_languages"]
    max_samples: int | None = args.max_samples or cfg.get("generation", {}).get("max_samples")
    skip_on_error: bool = cfg.get("generation", {}).get("skip_on_error", True)

    model_names: list[str] = args.models or enabled_models(cfg)
    if not model_names:
        logger.error("No models enabled.  Check config.yaml or pass --models.")
        sys.exit(1)

    output_dir = Path(cfg["output_dir"])

    # ── Dataset preparation ───────────────────────────────────────────── #
    ds_cfg = cfg["dataset"]
    processor = DatasetProcessor(
        dataset_name=ds_cfg["name"],
        split=ds_cfg.get("split", "dev"),
        text_columns=ds_cfg.get("text_columns", {}),
        output_dir=output_dir,
        target_languages=target_langs,
        max_samples=max_samples,
    )

    logger.info("Preparing dataset …")
    processor.prepare()

    # ── Dry-run exit ──────────────────────────────────────────────────── #
    if args.dry_run:
        sample_count = sum(1 for _ in processor.iter_samples())
        target_text_count = sum(len(lang_texts) for _, _, _, lang_texts in processor.iter_samples())
        logger.info(
            "DRY RUN — would process: %d samples × %d target texts × %d models",
            sample_count, target_text_count, len(model_names),
        )
        logger.info("Models: %s", model_names)
        logger.info("Languages: %s", target_langs)
        logger.info("Dry-run complete.  No audio generated.")
        return

    # ── Build model wrappers ──────────────────────────────────────────── #
    models_cfg = cfg["models"]
    model_instances = []
    for name in model_names:
        if name not in models_cfg:
            logger.warning("Model '%s' not found in config, skipping.", name)
            continue
        try:
            instance = build_model(name, models_cfg[name])
            model_instances.append(instance)
            logger.info("Registered model: %s → %s", name, instance)
        except Exception as exc:
            logger.error("Failed to build model '%s': %s", name, exc, exc_info=True)
            if not skip_on_error:
                raise

    if not model_instances:
        logger.error("No models could be instantiated.  Exiting.")
        sys.exit(1)

    # ── Run generation ────────────────────────────────────────────────── #
    pipeline = GenerationPipeline(
        models=model_instances,
        dataset_iter=processor.iter_samples(),
        output_dir=output_dir,
        target_langs=target_langs,
        skip_on_error=skip_on_error,
    )
    pipeline.run()

    logger.info("All done.  Outputs in: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
