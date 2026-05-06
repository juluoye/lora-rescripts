import argparse

from library import concept_edit_util, train_util
from library.utils import setup_logging
import sdxl_train_network

setup_logging()
import logging

logger = logging.getLogger(__name__)


class SdxlConceptEditNetworkTrainer(concept_edit_util.ConceptEditTrainerMixin, sdxl_train_network.SdxlNetworkTrainer):
    pass


def setup_parser() -> argparse.ArgumentParser:
    parser = sdxl_train_network.setup_parser()
    concept_edit_util.add_concept_edit_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    concept_edit_util.apply_concept_edit_runtime_defaults(args, logger)

    trainer = SdxlConceptEditNetworkTrainer()
    trainer.train(args)
