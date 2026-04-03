from pathlib import Path

from anima_gs_ghost.layout import sequence_layout


def test_sequence_layout_has_ghost_build() -> None:
    layout = sequence_layout(Path("artifacts"), "demo")
    assert "ghost_build" in layout
    assert layout["ghost_build"] == Path("artifacts/demo/ghost_build")


def test_sequence_layout_exposes_combined_output() -> None:
    layout = sequence_layout(Path("artifacts"), "demo")
    assert layout["combined_output"] == Path("artifacts/demo/output/combined")
