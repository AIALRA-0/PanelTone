import pytest

from manga_repaint.vlm_proposals import parse_material_labels, reviewed_materials


def test_vlm_material_labels_never_auto_accept_confidence() -> None:
    proposals = parse_material_labels(
        {
            "labels": [
                {"id": 0, "material": "skin", "confidence": 0.99, "reason": "guess"},
                {"id": 1, "material": "cloth", "confidence": 0.95},
            ]
        },
        expected_ids={0, 1},
    )
    assert reviewed_materials(proposals) == (None, None)


def test_reviewed_vlm_material_can_be_used_as_non_spatial_label() -> None:
    proposals = parse_material_labels(
        {
            "labels": [
                {
                    "id": 0,
                    "material": "skin",
                    "confidence": 0.9,
                    "reviewed": True,
                }
            ]
        },
        expected_ids={0},
    )
    assert reviewed_materials(proposals) == ("skin",)


def test_vlm_spatial_claims_are_rejected() -> None:
    with pytest.raises(ValueError, match="coordinates"):
        parse_material_labels(
            {
                "labels": [
                    {
                        "id": 0,
                        "material": "skin",
                        "confidence": 1.0,
                        "box": [0, 0, 10, 10],
                    }
                ]
            },
            expected_ids={0},
        )
