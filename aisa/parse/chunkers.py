from typing import Protocol, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from aisa.gen.prompts import get_token_count
from aisa.gen.chat_llm import BaseLLM
from aisa.utils import logger


class Chunker(Protocol):
    def split(self, text: str) -> list[str]: ...


def _shared_suffix_prefix_len(a: str, b: str, max_scan: int = 1200) -> int:
    max_len = min(len(a), len(b), max_scan)
    for n in range(max_len, 0, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


def _validate_split_response(resp, win_start: int, win_end: int) -> list[int]:
    if not isinstance(resp, dict) or not resp:
        return []
    raw = resp.get("split_after")
    if not isinstance(raw, list):
        return []
    valid: list[int] = []
    prev: int = win_start - 1
    for v in raw:
        if not isinstance(v, int) or isinstance(v, bool):
            continue
        if v < win_start or v >= win_end:
            continue
        if v <= prev:
            continue
        valid.append(v)
        prev = v
    return valid


def _llm_split_decisions(
    llm: BaseLLM,
    prompt_template: str,
    pieces: list[str],
    window: int,
    stride: int,
) -> list[int]:
    n: int = len(pieces)
    all_splits: set[int] = set()
    win_start: int = 0
    while win_start < n:
        win_end: int = min(win_start + window, n)
        if win_end - win_start <= 1:
            break
        tagged: str = "".join(
            f"<start_chunk_{i}>{pieces[i]}<end_chunk_{i}>"
            for i in range(win_start, win_end)
        )
        prompt: str = prompt_template.replace("{tagged_text}", tagged)
        try:
            resp = llm.query(prompt, verbose=False)
        except Exception as e:
            logger.log(
                "CHUNK",
                f"Logical chunker LLM call failed ({e}); splitting at window end {win_end - 1}",
            )
            all_splits.add(win_end - 1)
            if win_end >= n:
                break
            win_start += stride
            continue

        valid: list[int] = _validate_split_response(resp, win_start, win_end)
        if not valid:
            logger.log(
                "CHUNK",
                f"Logical chunker returned no valid splits for window [{win_start},{win_end}); splitting at window end",
            )
            valid = [win_end - 1]
        all_splits.update(valid)
        if win_end >= n:
            break
        win_start += stride

    return sorted(all_splits)


def _join_pieces(pieces: list[str], indices: list[int], has_overlap: bool) -> str:
    if not indices:
        return ""
    if not has_overlap or len(indices) == 1:
        return "".join(pieces[i] for i in indices)
    text: str = pieces[indices[0]]
    for idx in indices[1:]:
        ovl: int = _shared_suffix_prefix_len(text, pieces[idx])
        text += pieces[idx][ovl:]
    return text


def _assemble_with_overlap_trim(
    pieces: list[str],
    split_points: list[int],
    has_overlap: bool,
) -> tuple[list[str], list[list[int]]]:
    if not pieces:
        return [], []

    boundaries: list[int] = sorted(split_points)
    groups: list[list[int]] = []
    start: int = 0
    for sp in boundaries:
        groups.append(list(range(start, sp + 1)))
        start = sp + 1
    if start < len(pieces):
        groups.append(list(range(start, len(pieces))))

    chunks: list[str] = []
    sources: list[list[int]] = []
    for grp in groups:
        if not grp:
            continue
        text = _join_pieces(pieces, grp, has_overlap)
        if text.strip():
            chunks.append(text)
            sources.append(grp)
    return chunks, sources


def group_kept_pieces(
    pieces: list[str],
    kept_indices: list[int],
    llm: BaseLLM,
    prompt_template: str,
    window: int,
    stride: int,
    has_overlap: bool,
) -> tuple[list[str], list[list[int]]]:
    """Logical grouping over a masked subset of recursive pieces.

    Splits ``kept_indices`` into maximal contiguous runs, runs ``_llm_split_decisions``
    over each run independently, and concatenates the results. Gaps between runs are
    implicit hard splits — no logical chunk crosses a dropped piece. Returned
    ``source_chunk_ids`` reference original (unfiltered) piece indices.
    """
    if not kept_indices:
        return [], []

    runs: list[list[int]] = []
    current: list[int] = [kept_indices[0]]
    for idx in kept_indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            runs.append(current)
            current = [idx]
    runs.append(current)

    all_chunks: list[str] = []
    all_sources: list[list[int]] = []
    for run in runs:
        sub_pieces: list[str] = [pieces[i] for i in run]
        if len(sub_pieces) == 1:
            text = sub_pieces[0]
            if text.strip():
                all_chunks.append(text)
                all_sources.append([run[0]])
            continue
        sub_splits: list[int] = _llm_split_decisions(
            llm, prompt_template, sub_pieces, window, stride
        )
        sub_chunks, sub_sources = _assemble_with_overlap_trim(
            sub_pieces, sub_splits, has_overlap
        )
        for ch, src in zip(sub_chunks, sub_sources):
            all_chunks.append(ch)
            all_sources.append([run[i] for i in src])

    return all_chunks, all_sources


class RecursiveTextChunker:
    def __init__(self, chunk_size: int, recursive_overlap: int):
        self.chunk_size: int = chunk_size
        self.splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=recursive_overlap,
            length_function=get_token_count,
        )

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return self.splitter.split_text(text)


class LLMSemanticChunker:
    def __init__(
        self,
        llm: BaseLLM,
        chunk_size: int,
        recursive_overlap: int,
        logical_presplit_tokens: int = 50,
        logical_window: int = 40,
        logical_stride: int = 30,
    ):
        if logical_stride <= 0 or logical_stride > logical_window:
            raise ValueError(
                f"logical_stride ({logical_stride}) must be in (0, logical_window ({logical_window})]"
            )
        self.llm: BaseLLM = llm
        self.chunk_size: int = chunk_size
        self.window: int = logical_window
        self.stride: int = logical_stride
        self.presplitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=logical_presplit_tokens,
            chunk_overlap=0,
            length_function=get_token_count,
        )
        self.fallback_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=recursive_overlap,
            length_function=get_token_count,
        )
        self.prompt_template: str = llm.read_prompt("nemo_logical-chunk-02")

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        pieces: list[str] = self.presplitter.split_text(text)
        if len(pieces) <= 1:
            return pieces
        split_points: list[int] = _llm_split_decisions(
            self.llm, self.prompt_template, pieces, self.window, self.stride
        )
        chunks, _ = _assemble_with_overlap_trim(pieces, split_points, has_overlap=False)
        return self._enforce_size_cap(chunks)

    def _enforce_size_cap(self, chunks: list[str]) -> list[str]:
        cap: int = self.chunk_size * 2
        out: list[str] = []
        for ch in chunks:
            if get_token_count(ch) > cap:
                logger.log(
                    "CHUNK",
                    f"Logical chunk exceeded {cap} tokens; re-splitting with recursive fallback",
                )
                out.extend(self.fallback_splitter.split_text(ch))
            else:
                out.append(ch)
        return out


class HybridLogicalChunker:
    def __init__(
        self,
        llm: BaseLLM,
        chunk_size: int,
        recursive_overlap: int,
        hybrid_window: int = 8,
        hybrid_stride: int = 6,
    ):
        if hybrid_stride <= 0 or hybrid_stride > hybrid_window:
            raise ValueError(
                f"hybrid_stride ({hybrid_stride}) must be in (0, hybrid_window ({hybrid_window})]"
            )
        approx_input_tokens: int = hybrid_window * chunk_size
        if approx_input_tokens > llm.cfg.max_input_tokens:
            raise ValueError(
                f"hybrid_window ({hybrid_window}) × chunk_size ({chunk_size}) = {approx_input_tokens} "
                f"exceeds [llm].max_input_tokens ({llm.cfg.max_input_tokens})"
            )
        self.llm: BaseLLM = llm
        self.chunk_size: int = chunk_size
        self.recursive_overlap: int = recursive_overlap
        self.window: int = hybrid_window
        self.stride: int = hybrid_stride
        self.recursive: RecursiveTextChunker = RecursiveTextChunker(
            chunk_size, recursive_overlap
        )
        self.prompt_template: str = llm.read_prompt("nemo_logical-chunk")
        self.last_recursive_pieces: list[str] = []
        self.last_source_indices: list[list[int]] = []

    def split(self, text: str) -> list[str]:
        if not text.strip():
            self.last_recursive_pieces = []
            self.last_source_indices = []
            return []
        pieces: list[str] = self.recursive.split(text)
        self.last_recursive_pieces = pieces
        if len(pieces) <= 1:
            self.last_source_indices = [[i] for i in range(len(pieces))]
            return pieces
        split_points: list[int] = _llm_split_decisions(
            self.llm, self.prompt_template, pieces, self.window, self.stride
        )
        chunks, sources = _assemble_with_overlap_trim(
            pieces, split_points, has_overlap=self.recursive_overlap > 0
        )
        self.last_source_indices = sources
        return chunks


def get_chunker(chunk_cfg: dict, llm: Optional[BaseLLM] = None) -> Chunker:
    method: str = chunk_cfg.get("method", "recursive")
    if method == "recursive":
        return RecursiveTextChunker(
            chunk_size=chunk_cfg.get("chunk_size", 256),
            recursive_overlap=chunk_cfg.get("recursive_overlap", 50),
        )
    if method == "logical":
        if llm is None:
            raise ValueError("logical chunking requires a BaseLLM instance")
        return LLMSemanticChunker(
            llm=llm,
            chunk_size=chunk_cfg.get("chunk_size", 256),
            recursive_overlap=chunk_cfg.get("recursive_overlap", 50),
            logical_presplit_tokens=chunk_cfg.get("logical_presplit_tokens", 50),
            logical_window=chunk_cfg.get("logical_window", 40),
            logical_stride=chunk_cfg.get("logical_stride", 30),
        )
    if method == "random_logical":
        if llm is None:
            raise ValueError("random_logical chunking requires a BaseLLM instance")
        return HybridLogicalChunker(
            llm=llm,
            chunk_size=chunk_cfg.get("chunk_size", 256),
            recursive_overlap=chunk_cfg.get("recursive_overlap", 50),
            hybrid_window=chunk_cfg.get("hybrid_window", 8),
            hybrid_stride=chunk_cfg.get("hybrid_stride", 6),
        )
    raise ValueError(
        f"Unknown chunking method: {method!r}; expected one of 'recursive', 'logical', 'random_logical'"
    )
