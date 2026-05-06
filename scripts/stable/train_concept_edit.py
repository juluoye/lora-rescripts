import argparse

import train_network
from library import concept_edit_util, train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class ConceptEditNetworkTrainer(concept_edit_util.ConceptEditTrainerMixin, train_network.NetworkTrainer):
    pass


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    concept_edit_util.add_concept_edit_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    concept_edit_util.apply_concept_edit_runtime_defaults(args, logger)

    model_train_type = str(getattr(args, "model_train_type", "") or "").strip().lower()
    if model_train_type.startswith("sdxl-"):
        raise ValueError(
            "SDXL concept edit routes must use scripts/stable/sdxl_train_concept_edit.py, "
            "not scripts/stable/train_concept_edit.py."
        )

    trainer = ConceptEditNetworkTrainer()
    trainer.train(args)

