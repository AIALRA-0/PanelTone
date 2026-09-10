import numpy as np

from manga_repaint.sam2_regions import Sam2Region, deduplicate_regions


def _region(box, score=0.8):
    mask = np.zeros((100, 100), dtype=bool)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = True
    return Sam2Region(mask, score, ((left + right) // 2, (top + bottom) // 2))


def test_sam2_regions_reject_duplicate_masks_and_keep_best_score() -> None:
    lower = _region((10, 10, 50, 50), 0.75)
    better = _region((11, 11, 51, 51), 0.92)
    separate = _region((60, 60, 90, 90), 0.8)
    result = deduplicate_regions([lower, separate, better])
    assert result == (better, separate)


def test_sam2_region_validation_rejects_invalid_shape_and_point() -> None:
    region = _region((10, 10, 20, 20))
    region.validate((100, 100))
    try:
        region.validate((90, 90))
    except ValueError as error:
        assert "source-sized" in str(error)
    else:
        raise AssertionError("shape mismatch must fail")
