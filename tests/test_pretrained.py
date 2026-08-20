import pytest

from atlasfold import pretrained


@pytest.mark.parametrize(
    ("model_class", "runner_name"),
    [
        (pretrained.AtlasFold, "FoldingRunner"),
        (pretrained.AtlasFold_Multimer, "MultimerFoldingRunner"),
        (pretrained.AtlasFold_IPA, "IPAFoldingRunner"),
        (pretrained.AtlasFoldMultimer_IPA, "MultimerIPAFoldingRunner"),
    ],
)
def test_get_runner_dispatches_model(
    model_class: type,
    runner_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_class.__new__(model_class)
    expected_runner = object()

    monkeypatch.setattr(
        pretrained,
        runner_name,
        lambda loaded_model: expected_runner if loaded_model is model else None,
    )

    runner = pretrained.get_runner(model)

    assert runner is expected_runner


def test_get_runner_rejects_unknown_model_type() -> None:
    with pytest.raises(TypeError, match="Unsupported AtlasFold model type"):
        pretrained.get_runner(object())


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("atlasfold", pretrained.ATLASFOLD_260703),
        ("atlasfold-260703", pretrained.ATLASFOLD_260703),
        ("atlasfold-m", pretrained.ATLASFOLD_M_260725),
        ("atlasfold-m-260725", pretrained.ATLASFOLD_M_260725),
        ("atlasfold-ipa", pretrained.ATLASFOLD_IPA),
        ("atlasfold-multimer-ipa", pretrained.ATLASFOLD_MULTIMER_IPA),
    ],
)
def test_get_model_name_accepts_short_and_versioned_names(
    model_name: str,
    expected: str,
) -> None:
    assert pretrained.get_model_name(model_name) == expected
