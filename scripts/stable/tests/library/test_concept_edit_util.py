from argparse import Namespace

from library.concept_edit_util import infer_concept_edit_mode_from_training_type, normalize_concept_edit_mode


def test_infer_concept_edit_mode_from_training_type():
    assert infer_concept_edit_mode_from_training_type("sd-ileco") == "ileco"
    assert infer_concept_edit_mode_from_training_type("sd-addift") == "addift"
    assert infer_concept_edit_mode_from_training_type("sd-multi-addift") == "multi-addift"
    assert infer_concept_edit_mode_from_training_type("sdxl-multi-addift") == "multi-addift"
    assert infer_concept_edit_mode_from_training_type("anima-multi-addift") == "multi-addift"


def test_normalize_concept_edit_mode_accepts_aliases():
    assert normalize_concept_edit_mode("leco") == "ileco"
    assert normalize_concept_edit_mode("multi_addift") == "multi-addift"
    assert normalize_concept_edit_mode(None, "sdxl-addift") == "addift"
