import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from anima_lora_trainer_with_cooldown import setup_parser as setup_anima_parser
from library import anima_concept_edit_util, train_util
from library.utils import setup_logging
from lulynx.anima_lora_trainer import AnimaNetworkTrainer as BaseAnimaNetworkTrainer

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaConceptEditNetworkTrainer(
    anima_concept_edit_util.AnimaConceptEditTrainerMixin,
    BaseAnimaNetworkTrainer,
):
    pass


def setup_parser() -> argparse.ArgumentParser:
    parser = setup_anima_parser()
    anima_concept_edit_util.add_concept_edit_arguments(parser)
    return parser


def _restore_missing_parser_defaults(parser, args):
    for action in getattr(parser, "_actions", []):
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        if not hasattr(args, dest):
            setattr(args, dest, action.default)
    return args


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    args = _restore_missing_parser_defaults(parser, args)
    anima_concept_edit_util.apply_anima_concept_edit_runtime_defaults(args, logger)

    trainer = AnimaConceptEditNetworkTrainer()
    trainer.train(args)
