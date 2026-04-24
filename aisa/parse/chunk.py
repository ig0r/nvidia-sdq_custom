import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from rapidfuzz import process, fuzz
from pydantic import BaseModel, Field, field_validator
from aisa.utils import files, dictlist, logger
from aisa.gen.prompts import get_token_count
from aisa.gen.embed import EmbedConfig
from aisa.parse.doc import ParsedDoc

# TODO: ADD EQUATIONS? CREATE DIRECTORY FOR EXTR. IMAGES
MD_PATTERNS: dict[str, re.Pattern] = {
    "table": re.compile(
        r"(?:\|.*\|.*\n)"  # Header row
        r"(?:\|[\s:\-]+\|.*\n)"  # Separator row
        r"(?:\|.*\|.*\n?)+",  # Data rows
        flags=re.MULTILINE,
    ),
    "image": re.compile(r"!\[.*?\]\((.*?)\)"),
}


class Chunk(BaseModel):
    docid: str = Field(alias="docid", default="")
    doctitle: str = Field(alias="doctitle", default="")
    org: str = Field(alias="org", default="")
    year: str = Field(alias="year", default="")
    doctype: str = Field(alias="doctype", default="")
    categories: list[str] = Field(alias="categories", default=[])
    section: str = Field(alias="section", default="")
    subsection: str = Field(alias="subsection", default="")
    parsed_file: str = Field(alias="parsed_file", default="")
    pdf: str = Field(alias="pdf", default="")
    texts: list[str] = Field(alias="texts", default=[])
    images: list[str] = Field(alias="images", default=[])
    tables: list[str] = Field(alias="tables", default=[])


class Chunker:
    def __init__(self, embed_cfg: EmbedConfig):
        self.embed_cfg = embed_cfg
        self.splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=embed_cfg.chunk_size,
            chunk_overlap=embed_cfg.recursive_overlap,
            length_function=get_token_count,
        )

    def aggregate(self, doc: ParsedDoc, overwrite: bool = False) -> None:
        all_positions: list[dict[str, any]] = []
        chunks: list[Chunk] = []
        tables: list[str] = []
        images: list[str] = []
        temp_pos: dict[str, any] = {}
        last_pos: int = 0
        pos1: int = 0
        pos2: int = 0
        raw_content: str = "\n".join(
            [item for item in doc.content.split("\n") if item.strip() != ""]
        )
        out_path: str = f"{doc.base_out}-chunks.json"

        # check exists
        if files.exists(out_path) and not overwrite:
            logger.opt(colors=True).log(
                "CHUNK",
                f"`{doc.doctitle} ({doc.section_name})` already chunked:"
                + f" <yellow>{doc.base_out}-chunks.json</yellow>",
            )
            return

        logger.log("CHUNK", f"Chunking {doc.doctitle} ({doc.section_name})")

        # get positions of all subsections in the raw content
        if len(list(doc.all_subs.keys())) == len(list(doc.subsections.keys())):
            for key in doc.all_subs.keys():
                last_pos = 0
                try:
                    if len(all_positions) == 0:
                        temp_pos = fuzz_position(raw_content, doc.all_subs[key])
                    else:
                        last_pos = (
                            all_positions[-1]["position"] + all_positions[-1]["len"]
                        )
                        temp_pos = fuzz_position(
                            raw_content[last_pos:], doc.all_subs[key]
                        )
                        temp_pos["position"] += last_pos
                except Exception as e:
                    logger.error(f"Error processing subsection {key}: {e}")
                all_positions.append(temp_pos)
        else:
            all_positions.append({"position": 0, "len": 0})

        # if no positions found, set to start of document
        if all_positions == []:
            all_positions.append({"position": 0, "len": 0})
            doc.subsections = {"": "Unknown"}

        uk_name: str = ""
        if "#" in doc.doctitle:
            uk_name = doc.doctitle.split("#")[-1].strip()
            doc.doctitle = doc.doctitle.split("#")[0].strip()

        # get split chunks based on subsection positions
        for idx, key in enumerate(doc.subsections.keys()):
            pos1 = all_positions[idx]["position"] + all_positions[idx]["len"]
            pos2 = (
                all_positions[idx + 1]["position"]
                if idx + 1 < len(all_positions)
                else len(raw_content)
            )
            raw_section: str = raw_content[pos1:pos2]

            # extract and replace tables and images (md)
            tables = MD_PATTERNS["table"].findall(raw_section)
            images = list(set(MD_PATTERNS["image"].findall(raw_section)))
            raw_section = MD_PATTERNS["table"].sub("", raw_section)
            raw_section = MD_PATTERNS["image"].sub("", raw_section)
            chunks.append(
                Chunk(
                    **{
                        "docid": f"{doc.name.id}_{doc.name.chapter}",
                        "doctitle": doc.doctitle,
                        "org": doc.name.source,
                        "year": doc.year,
                        "doctype": doc.name.doctype_name,
                        "categories": doc.name.catnames,
                        "section": (
                            f"{doc.section_no} = {doc.section_name}"
                            if doc.section_name != "Unknown"
                            else uk_name
                        ),
                        "subsection": (
                            f"{key} = {doc.subsections[key]}"
                            if doc.subsections[key] != "Unknown"
                            else ""
                        ),
                        "parsed_file": f"{doc.parent_dir}/{doc.name.filename}".replace(
                            "\\", "/"
                        ),
                        "pdf": doc.pdf_path.replace("\\", "/"),
                        "texts": self.splitter.split_text(raw_section),
                        "images": images,
                        "tables": tables,
                    }
                )
            )

        # export
        with open(out_path, "w") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=4)
        logger.opt(colors=True).log(
            "CHUNK",
            f"Exported chunks: <yellow>{out_path}</yellow>",
        )


# NOTE: DEPRECATED
# def fuzz_position(text: str, target: str):
#     target_len = len(target) + 10
#     substrings = [text[i : i + target_len] for i in range(len(text) - target_len + 1)]
#     best_match, score, index = process.extractOne(target, substrings, scorer=fuzz.ratio)
#     return {
#         "position": index,
#         "match": best_match,
#         "score": score,
#         "len": len(best_match),
#     }


def fuzz_position(text: str, target: str, window_pad: int = 10, min_score: int = 70):
    # --- 1. Fast path: exact search
    text = text.upper()
    target = target.upper()
    pos = text.find(target)
    if pos != -1:
        return {
            "position": pos,
            "match": target,
            "score": 100,
            "len": len(target),
        }

    # --- 2. Fuzzy matching with extract (multiple candidates)
    target_len = len(target)
    substrings = (
        text[i : i + target_len + window_pad]
        for i in range(len(text) - target_len - window_pad + 1)
    )

    candidates = process.extract(
        target, substrings, scorer=fuzz.partial_ratio, limit=5  # keep top 5 matches
    )

    if candidates:
        best_match, score, index = max(
            candidates, key=lambda x: x[1]
        )  # pick highest score
        if score >= min_score:
            return {
                "position": index,
                "match": best_match,
                "score": score,
                "len": len(best_match),
            }

    # --- 3. No reliable match
    return {
        "position": 0,
        "match": "",
        "score": 0,
        "len": 0,
    }


def reduce_token_count(
    chunks: list[str],
    custom_input: dictlist,
    template: PromptTemplate = None,
    max_tokens: int = 4000,
) -> dictlist:
    prompt = ""
    if custom_input is None:
        if template != None:
            if isinstance(chunks[0], str):
                prompt = template.format(text="\n".join(chunks))
            elif isinstance(chunks[0], dict):
                prompt = template.format(text=json.dumps(chunks))
        else:
            prompt = "\n".join(chunks)
    else:
        prompt = "\n".join(template.format(**item) for item in chunks)

    token_count: int = get_token_count(prompt)
    if token_count > max_tokens:
        chunks.pop()
        return reduce_token_count(chunks, custom_input, template, max_tokens)
    else:
        return chunks


def RecursiveChunker(
    chunks: list[str] = [],
    custom_input: dictlist = None,
    prompt: str = "",
    max_input_tokens: int = 4000,
    max_chunk_size: int = 20,
    return_type: type = str,
) -> dictlist:
    if chunks == [] and custom_input is not None:
        chunks = [json.dumps(item) for item in custom_input]
    elif chunks == [] and custom_input is None:
        return []

    i: int = 0
    outputs: dictlist = []
    template: PromptTemplate = None

    if prompt != "":
        if custom_input is None:
            template = PromptTemplate(template=prompt, input_variables=["text"])
        else:
            template = PromptTemplate(
                template=prompt, input_variables=custom_input[0].keys()
            )
    while i < len(chunks):
        sample_chunks: dictlist = []
        if custom_input is not None:
            sample_chunks = custom_input[i : i + max_chunk_size]
        else:
            sample_chunks = chunks[i : i + max_chunk_size]

        reduce_chunks: dictlist = reduce_token_count(
            chunks=sample_chunks,
            custom_input=custom_input,
            template=template,
            max_tokens=max_input_tokens,
        )
        if len(chunks) > max_chunk_size:
            chunks = chunks + sample_chunks[len(reduce_chunks) :]

        if reduce_chunks == []:
            break
        if return_type == str:
            outputs.append(",\n".join(reduce_chunks))
        else:
            outputs.append(reduce_chunks)
        i += len(reduce_chunks)
    return outputs
