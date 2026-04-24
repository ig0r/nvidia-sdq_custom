from typing import Protocol, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from aisa.gen.prompts import get_token_count
from aisa.gen.chat_llm import BaseLLM
from aisa.utils import logger


class Chunker(Protocol):
    def split(self, text: str) -> list[str]: ...


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
        self.prompt_template: str = llm.read_prompt("nemo_logical-chunk")

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        pieces: list[str] = self.presplitter.split_text(text)
        if len(pieces) <= 1:
            return pieces
        split_points: list[int] = self._get_split_points(pieces)
        chunks: list[str] = self._assemble(pieces, split_points)
        return self._enforce_size_cap(chunks)

    def _get_split_points(self, pieces: list[str]) -> list[int]:
        n: int = len(pieces)
        all_splits: set[int] = set()
        win_start: int = 0
        while win_start < n:
            win_end: int = min(win_start + self.window, n)
            if win_end - win_start <= 1:
                break
            tagged: str = "".join(
                f"<start_chunk_{i}>{pieces[i]}<end_chunk_{i}>"
                for i in range(win_start, win_end)
            )
            prompt: str = self.prompt_template.replace("{tagged_text}", tagged)
            try:
                resp = self.llm.query(prompt, verbose=False)
            except Exception as e:
                logger.log(
                    "CHUNK",
                    f"Logical chunker LLM call failed ({e}); splitting at window end {win_end - 1}",
                )
                all_splits.add(win_end - 1)
                if win_end >= n:
                    break
                win_start += self.stride
                continue

            valid: list[int] = self._validate_response(resp, win_start, win_end)
            if not valid:
                logger.log(
                    "CHUNK",
                    f"Logical chunker returned no valid splits for window [{win_start},{win_end}); splitting at window end",
                )
                valid = [win_end - 1]
            all_splits.update(valid)
            if win_end >= n:
                break
            win_start += self.stride

        return sorted(all_splits)

    def _validate_response(self, resp, win_start: int, win_end: int) -> list[int]:
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

    def _assemble(self, pieces: list[str], split_points: list[int]) -> list[str]:
        if not split_points:
            return ["".join(pieces)]
        chunks: list[str] = []
        start: int = 0
        for sp in split_points:
            chunks.append("".join(pieces[start : sp + 1]))
            start = sp + 1
        if start < len(pieces):
            chunks.append("".join(pieces[start:]))
        return [c for c in chunks if c.strip()]

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
    raise ValueError(
        f"Unknown chunking method: {method!r}; expected one of 'recursive', 'logical'"
    )
