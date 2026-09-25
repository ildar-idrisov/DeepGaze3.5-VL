import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.special import logsumexp

from evaluate_vllm_unified import (
    INTERNVL_DEFAULT_SYSTEM,
    FewShotPromptBuilder,
    _check_merged_dtype,
    load_centerbias_from_pkl,
    summarize_results,
)

DATA = Path(__file__).resolve().parents[1] / 'data'
PROMPT = "Analyze this image and predict a human eye movement scanpath."


def test_prompt_reproduces_llamafactory_intern_vl_training_format():
    # LlamaFactory `intern_vl`: default system prompt, then <image> replaced in place by the
    # image tokens and directly followed by the text (vLLM expands <IMG_CONTEXT> the same way)
    image = Image.new('RGB', (8, 8))
    prompt, mm_data = FewShotPromptBuilder().build_prompt(image, PROMPT, [], partial_response="[(")
    assert prompt == (
        f"<|im_start|>system\n{INTERNVL_DEFAULT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n<IMG_CONTEXT>{PROMPT}<|im_end|>\n"
        f"<|im_start|>assistant\n[("
    )
    assert mm_data == {"image": image}


def test_prompt_with_shots_keeps_one_system_turn_and_orders_images():
    test_image, shot_image = Image.new('RGB', (8, 8)), Image.new('RGB', (4, 4))
    shots = [{'image': shot_image, 'prompt': PROMPT, 'response': "[(50, 50), (10, 20)]"}]
    prompt, mm_data = FewShotPromptBuilder().build_prompt(test_image, PROMPT, shots)
    assert prompt.count("<|im_start|>system") == 1
    assert prompt.count("<IMG_CONTEXT>") == 2
    assert "<|im_start|>assistant\n[(50, 50), (10, 20)]<|im_end|>\n<|im_start|>user\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert mm_data == {"image": [shot_image, test_image]}


def test_centerbias_is_loaded_and_normalized_on_the_grid():
    log_density, pkl_path = load_centerbias_from_pkl('images/MIT_0985.jpg', str(DATA / 'centerbias'))
    assert log_density.shape == (100, 100)
    assert logsumexp(log_density) == pytest.approx(0.0, abs=1e-6)
    assert pkl_path.endswith('MIT/0985.pkl')


def test_missing_centerbias_raises_instead_of_using_synthetic_prior():
    with pytest.raises(FileNotFoundError):
        load_centerbias_from_pkl('images/MIT_9999.jpg', str(DATA / 'centerbias'))


def test_summary_weights_fixations_images_and_scanpaths():
    results = [
        {'image': 'a', 'lp_fixation_igs': [1.0], 'lp_fixation_lls': [-9.0]},
        {'image': 'a', 'lp_fixation_igs': [3.0, 3.0, 3.0], 'lp_fixation_lls': [-7.0, -7.0, -7.0]},
        {'image': 'b', 'lp_fixation_igs': [0.0], 'lp_fixation_lls': [-10.0]},
    ]
    ig = summarize_results(results)['IG (bits)']
    assert ig['per_fixation'] == pytest.approx(2.0)        # (1 + 9 + 0) / 5
    assert ig['per_image'] == pytest.approx(1.25)          # mean(2.5, 0)
    assert ig['per_scanpath'] == pytest.approx(4.0 / 3.0)  # mean(1, 3, 0)
    assert (ig['num_fixations'], ig['num_images'], ig['num_scanpaths']) == (5, 2, 3)


def test_summary_of_grid_mode_results():
    metrics = [{'ig': 1.0, 'll': -8.0, 'auc': 0.9, 'nss': 2.0, 'log_nss': 1.0},
               {'ig': 0.0, 'll': -9.0, 'auc': 0.7, 'nss': 1.0, 'log_nss': 0.5}]
    summary = summarize_results([{'image': 'a', 'lp_fixation_metrics': metrics}])
    assert summary['AUC']['per_fixation'] == pytest.approx(0.8)
    assert summary['NSS']['per_fixation'] == pytest.approx(1.5)


@pytest.mark.parametrize('key', ['torch_dtype', 'dtype'])
def test_merged_model_in_float16_is_not_reused(tmp_path, key):
    (tmp_path / 'config.json').write_text(json.dumps({key: 'float16'}))
    with pytest.raises(RuntimeError):
        _check_merged_dtype(str(tmp_path))
    (tmp_path / 'config.json').write_text(json.dumps({key: 'bfloat16'}))
    _check_merged_dtype(str(tmp_path))
