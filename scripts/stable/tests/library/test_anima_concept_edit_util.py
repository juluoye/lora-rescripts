from library.anima_concept_edit_util import infer_anima_concept_edit_mode_from_training_type


def test_infer_anima_concept_edit_mode_from_training_type():
    assert infer_anima_concept_edit_mode_from_training_type("anima-ileco") == "ileco"
    assert infer_anima_concept_edit_mode_from_training_type("anima-addift") == "addift"
    assert infer_anima_concept_edit_mode_from_training_type("anima-multi-addift") == "multi-addift"
