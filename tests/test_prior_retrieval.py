from pathlib import Path

import pytest

from anima_gs_ghost.preprocess.prior_retrieval import PriorIndexEntry, PriorRetriever, SimplePriorIndex


def test_prior_retrieval_accepts_explicit_and_vlm_prompts(tmp_path: Path) -> None:
    index = SimplePriorIndex(
        entries=(
            PriorIndexEntry(
                uid="mug-001",
                name="Ceramic Mug",
                description="cylindrical cup with a side handle",
                mesh_path=tmp_path / "meshes" / "mug.glb",
            ),
            PriorIndexEntry(
                uid="kettle-001",
                name="Metal Kettle",
                description="rounded spout body and top handle",
                mesh_path=tmp_path / "meshes" / "kettle.glb",
            ),
        )
    )

    result = PriorRetriever(index=index).retrieve(
        prompt="ceramic cup with handle",
        vlm_prompt="mug, drinking cup",
        sequence="demo-seq",
        data_root=tmp_path,
        topk=2,
    )

    assert [prompt.source for prompt in result.prompts] == ["explicit", "vlm"]
    assert len(result.candidates) == 1
    assert result.candidates[0].uid == "mug-001"
    assert result.output_dir == tmp_path / "demo-seq" / "ghost_build" / "openshape"
    assert result.candidates[0].simplified_mesh_path.name.startswith("00_ceramic_mug")


def test_prior_retrieval_limits_and_orders_topk(tmp_path: Path) -> None:
    index = SimplePriorIndex(
        entries=(
            PriorIndexEntry(uid="a", name="Laptop", description="thin metal laptop computer"),
            PriorIndexEntry(uid="b", name="Notebook", description="paper notebook sketch book"),
            PriorIndexEntry(uid="c", name="Microwave", description="kitchen microwave oven"),
        )
    )

    result = PriorRetriever(index=index).retrieve(
        prompt=("metal laptop", "computer"),
        topk=1,
        sequence="demo-seq",
        data_root=tmp_path,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].uid == "a"
    assert result.candidates[0].ready_for_alignment


def test_prior_retrieval_requires_any_prompt_source() -> None:
    with pytest.raises(ValueError, match="requires at least one explicit or VLM-generated prompt"):
        PriorRetriever().retrieve()
