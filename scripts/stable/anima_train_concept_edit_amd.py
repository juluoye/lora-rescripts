import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from anima_train_network_amd import _restore_missing_parser_defaults
from anima_lora_trainer_with_cooldown import setup_parser as setup_anima_parser
from library import anima_concept_edit_util, train_util
from lulynx_amd.anima_lora_trainer import AnimaNetworkTrainer as BaseAmdAnimaNetworkTrainer


class AnimaConceptEditNetworkTrainer(
    anima_concept_edit_util.AnimaConceptEditTrainerMixin,
    BaseAmdAnimaNetworkTrainer,
):
    pass


def setup_parser():
    parser = setup_anima_parser()
    anima_concept_edit_util.add_concept_edit_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    args = _restore_missing_parser_defaults(parser, args)
    anima_concept_edit_util.apply_anima_concept_edit_runtime_defaults(args, __import__("logging").getLogger(__name__))

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    trainer = AnimaConceptEditNetworkTrainer()
    trainer.train(args)
