from __future__ import annotations

import numpy as np

from manga_repaint.semantic import _validated_bubble_regions


def test_semantic_bubbles_reject_unsupported_bright_body_region() -> None:
    trusted = np.zeros((80, 120), dtype=bool)
    trusted[10:35, 8:38] = True
    predicted = trusted.copy()
    predicted[45:60, 70:105] = True
    text = np.zeros_like(trusted)
    text[17:21, 15:20] = True
    text[25:29, 25:30] = True
    # A single face-like mark must not validate the separate bright region.
    text[50:53, 84:88] = True

    result = _validated_bubble_regions(trusted, predicted, text)

    assert result[20, 20]
    assert not result[52, 90]


def test_semantic_bubbles_accept_model_region_with_multiple_text_components() -> None:
    trusted = np.zeros((80, 120), dtype=bool)
    predicted = np.zeros_like(trusted)
    predicted[20:60, 45:95] = True
    text = np.zeros_like(trusted)
    text[30:34, 58:63] = True
    text[42:46, 72:77] = True

    result = _validated_bubble_regions(trusted, predicted, text)

    assert result[40, 70]
