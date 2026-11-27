"""`scripts/update_openrouter_input_modalities.py`, which writes the snapshot `OpenRouterProvider` reads.

The provider falls back to a variant's base model for an ID the snapshot doesn't list, so the script may drop a `:tag`
variant only when it takes the same kinds as its base. These tests pin that rule from the script's side, with a fixed
model list instead of OpenRouter's.
"""

from __future__ import annotations as _annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / 'scripts'))

from update_openrouter_input_modalities import ModelList, input_modalities, render


def _model_list(*models: tuple[str, list[str] | None]) -> ModelList:
    return ModelList.model_validate(
        {
            'data': [
                {'id': model_id} if kinds is None else {'id': model_id, 'architecture': {'input_modalities': kinds}}
                for model_id, kinds in models
            ]
        }
    )


class TestInputModalities:
    def test_variant_like_its_base_is_left_to_the_fallback(self):
        listed = _model_list(('acme/vision', ['text', 'image']), ('acme/vision:free', ['image', 'text']))
        assert input_modalities(listed) == {'acme/vision': ('image',)}

    def test_variant_unlike_its_base_is_kept(self):
        listed = _model_list(('acme/vision', ['text', 'image']), ('acme/vision:batch', ['text']))
        assert input_modalities(listed) == {'acme/vision': ('image',), 'acme/vision:batch': ()}

    def test_variant_without_a_listed_base_is_kept(self):
        assert input_modalities(_model_list(('acme/only-free:free', ['text', 'audio']))) == {
            'acme/only-free:free': ('audio',)
        }

    def test_entry_without_input_modalities_is_skipped(self):
        listed = _model_list(('acme/unknown', None), ('acme/text-only', ['text', 'file']))
        assert input_modalities(listed) == {'acme/text-only': ()}

    def test_model_id_that_could_break_out_of_the_snapshot_is_rejected(self):
        with pytest.raises(ValueError, match='Unexpected model ID'):
            input_modalities(_model_list(("acme/x'] = 1 #", ['text'])))


def test_render_is_sorted_python_with_one_tuple_per_model():
    source = render({'b/omni': ('image', 'audio', 'video'), 'a/vision': ('image',), 'a/text': ()})
    namespace: dict[str, object] = {}
    exec(compile(source, 'snapshot', 'exec'), namespace)
    assert namespace['OPENROUTER_INPUT_MODALITIES'] == {
        'a/text': (),
        'a/vision': ('image',),
        'b/omni': ('image', 'audio', 'video'),
    }
    assert source.index("'a/text'") < source.index("'a/vision'") < source.index("'b/omni'")
