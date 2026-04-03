"""Prompt and prior-retrieval contracts for paper-faithful geometric priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal, Protocol

from anima_gs_ghost.config import GhostSettings
from anima_gs_ghost.layout import SequenceLayout

PromptSource = Literal["explicit", "vlm"]


def _slugify(value: str) -> str:
    value = re.sub(r"[^\w\s-]", "", value.strip().lower())
    value = re.sub(r"\s+", "_", value)
    return value[:64] if value else "prior"


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 1}


@dataclass(frozen=True)
class PriorPrompt:
    text: str
    source: PromptSource


@dataclass(frozen=True)
class PriorIndexEntry:
    uid: str
    name: str
    description: str
    mesh_path: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PriorCandidate:
    rank: int
    uid: str
    name: str
    score: float
    retrieved_mesh_path: Path | None
    simplified_mesh_path: Path
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def ready_for_alignment(self) -> bool:
        return self.retrieved_mesh_path is not None or self.simplified_mesh_path.suffix == ".obj"


@dataclass(frozen=True)
class PriorRetrievalResult:
    prompts: tuple[PriorPrompt, ...]
    candidates: tuple[PriorCandidate, ...]
    output_dir: Path
    retrieval_backend: str


class PromptProducer(Protocol):
    def generate(self, sequence: str, image_dir: Path) -> str:
        """Generate a VLM-style prompt for the given sequence."""


class PriorIndex(Protocol):
    def search(self, prompts: tuple[PriorPrompt, ...], topk: int, output_dir: Path) -> tuple[PriorCandidate, ...]:
        """Return ranked prior candidates for the provided prompts."""


class SimplePriorIndex:
    """Keyword-overlap index used for local scaffolding and tests."""

    def __init__(self, entries: tuple[PriorIndexEntry, ...], minimum_score: float = 2.0) -> None:
        self.entries = entries
        self.minimum_score = minimum_score

    def search(
        self,
        prompts: tuple[PriorPrompt, ...],
        topk: int,
        output_dir: Path,
    ) -> tuple[PriorCandidate, ...]:
        prompt_tokens = set().union(*(_tokenize(prompt.text) for prompt in prompts))
        scored: list[tuple[float, PriorIndexEntry]] = []
        for entry in self.entries:
            haystack = f"{entry.name} {entry.description} {' '.join(entry.metadata.values())}"
            entry_tokens = _tokenize(haystack)
            overlap = len(prompt_tokens & entry_tokens)
            if overlap < self.minimum_score:
                continue
            scored.append((float(overlap), entry))

        scored.sort(key=lambda item: (-item[0], item[1].name))
        candidates = []
        for rank, (score, entry) in enumerate(scored[:topk]):
            stem = _slugify(entry.name)
            candidates.append(
                PriorCandidate(
                    rank=rank,
                    uid=entry.uid,
                    name=entry.name,
                    score=score,
                    retrieved_mesh_path=entry.mesh_path,
                    simplified_mesh_path=output_dir / f"{rank:02d}_{stem}.obj",
                    metadata=entry.metadata,
                )
            )
        return tuple(candidates)


class PriorRetriever:
    """Resolve prompt sources and return ranked candidates ready for alignment."""

    def __init__(
        self,
        settings: GhostSettings | None = None,
        index: PriorIndex | None = None,
        prompt_producer: PromptProducer | None = None,
    ) -> None:
        self.settings = settings or GhostSettings()
        self.index = index or SimplePriorIndex(entries=())
        self.prompt_producer = prompt_producer

    def retrieve(
        self,
        prompt: str | tuple[str, ...] | None = None,
        *,
        vlm_prompt: str | tuple[str, ...] | None = None,
        topk: int | None = None,
        sequence: str | None = None,
        data_root: Path | None = None,
    ) -> PriorRetrievalResult:
        prompts = self._resolve_prompts(
            prompt=prompt,
            vlm_prompt=vlm_prompt,
            sequence=sequence,
            data_root=data_root,
        )
        if not prompts:
            raise ValueError("PriorRetriever requires at least one explicit or VLM-generated prompt.")

        target_topk = topk or self.settings.pipeline.prior_topk
        output_dir = self._output_dir(sequence=sequence, data_root=data_root, prompts=prompts)
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates = self.index.search(prompts=prompts, topk=target_topk, output_dir=output_dir)

        return PriorRetrievalResult(
            prompts=prompts,
            candidates=candidates,
            output_dir=output_dir,
            retrieval_backend=self.index.__class__.__name__,
        )

    def _resolve_prompts(
        self,
        *,
        prompt: str | tuple[str, ...] | None,
        vlm_prompt: str | tuple[str, ...] | None,
        sequence: str | None,
        data_root: Path | None,
    ) -> tuple[PriorPrompt, ...]:
        resolved: list[PriorPrompt] = []
        resolved.extend(self._normalize_prompts(prompt, source="explicit"))
        resolved.extend(self._normalize_prompts(vlm_prompt, source="vlm"))

        if resolved:
            return tuple(resolved)

        if self.prompt_producer is None or sequence is None or data_root is None:
            return ()

        layout = SequenceLayout(root=data_root, sequence=sequence)
        generated = self.prompt_producer.generate(sequence=sequence, image_dir=layout.ghost_build / "obj_rgb")
        return tuple(self._normalize_prompts(generated, source="vlm"))

    def _normalize_prompts(
        self,
        raw: str | tuple[str, ...] | None,
        *,
        source: PromptSource,
    ) -> list[PriorPrompt]:
        if raw is None:
            return []
        values = (raw,) if isinstance(raw, str) else raw
        return [PriorPrompt(text=value.strip(), source=source) for value in values if value.strip()]

    def _output_dir(
        self,
        *,
        sequence: str | None,
        data_root: Path | None,
        prompts: tuple[PriorPrompt, ...],
    ) -> Path:
        if sequence is not None and data_root is not None:
            layout = SequenceLayout(root=data_root, sequence=sequence)
            return layout.ghost_build / "openshape"
        stem = _slugify(prompts[0].text)
        return self.settings.data.artifacts_root / "priors" / stem
