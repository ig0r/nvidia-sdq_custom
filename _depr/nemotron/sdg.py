import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize

from aisa.utils import files
from aisa.gen.chat_llm import BaseLLM, LLMConfig

# Ensure NLTK punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# PROMPTS (from NVIDIA retriever-sdg NeMo Data Designer)
# ---------------------------------------------------------------------------

ARTIFACTS_SYSTEM_PROMPT = (
    "You are an expert at analyzing documents and extracting semantic artifacts."
)

ARTIFACTS_PROMPT = """Analyze the following content and extract semantic artifacts that would be valuable for generating high-quality question-answer pairs.

Note: The content may contain multiple documents bundled together (separated by "=== Document Boundary ==="). 
If multiple documents are present, identify cross-document relationships and connections.

CONTENT:
{text}

ARTIFACT TYPES TO EXTRACT:
- key_concepts: Core ideas and concepts discussed in the document(s)
- relationships: Connections and relationships between different concepts (including cross-document relationships)
- themes: Overarching themes and topics
- entities: Specific entities, people, organizations, or items mentioned
- processes: Processes, workflows, or procedures described
- insights: Key insights, conclusions, or findings
- technical_terms: Technical terminology and specialized vocabulary
- contextual_factors: Contextual information that provides background

INSTRUCTIONS:
1. Extract up to {max_artifacts} artifacts for each relevant type
2. Focus on the most significant and informative elements
3. Provide clear, concise descriptions for each artifact
4. Include context about why each artifact is important
5. Ensure artifacts are specific and actionable for Q&A generation
6. For multi-document bundles, pay special attention to relationships and comparisons between documents

Return output as a JSON object with lists of objects for each artifact type (keys: "key_concepts", "relationships", "themes", "entities", "processes", "insights", "technical_terms", "contextual_factors"). Each list element should have "text", "description", and "importance".
"""

QA_GEN_SYSTEM_PROMPT = "You are an expert at extracting question and answer pairs from provided context/transcript/segments."

QA_GEN_PROMPT = """You are an expert at extracting question and answer pairs from provided context/transcript/segments.

<document_facts_block>:
{facts_block}
</document_facts_block>

<context_block>:
{context_block}
</context_block>

Guidelines:
1. Generate questions with varying complexity levels between 1 (simple) and 5 (complex):
   - All questions MUST require understanding connections between different parts of the context/transcript/segments
   - Questions should test deep understanding, not simple facts
   - Do not mention the existence of a context/transcript in the generated question like "in the transcript", "from the given context", or "in Segment 148". Produce a natural, standalone question.
   - Only use facts present in the provided context/transcript; if missing, say you cannot generate a question.
   - Example: "How does the speaker's initial explanation of X relate to the later implementation of Y?"

2. Question Types to Generate (for the "query_type" field - ONLY these 3 values allowed):
   - "multi_hop" ({query_counts_multi_hop} questions): Connect {min_hops}-{max_hops} separated segments
   - "structural" ({query_counts_structural} questions): Focus on relationships between concepts
   - "contextual" ({query_counts_contextual} questions): Require surrounding context to understand
   - Use the cross-part context snippets to connect evidence that lives outside the current transcript section

3. Reasoning Types to Include (for the "reasoning_type" field - ONLY these 7 values allowed):
   - "factual" ({reasoning_counts_factual} questions): Ask for complex facts that require synthesizing multiple pieces of information (NOT simple lookups)
   - "relational" ({reasoning_counts_relational} questions): Ask how data points compare or correlate across different segments
   - "inferential" ({reasoning_counts_inferential} questions): Ask about conclusions or implications requiring synthesis
   - "temporal" ({reasoning_counts_temporal} questions): Ask about changes or events over time across segments
   - "procedural" ({reasoning_counts_procedural} questions): Ask about complex multi-step processes or guidelines
   - "visual" ({reasoning_counts_visual} questions): Ask about visual details requiring cross-reference
   - "causal" ({reasoning_counts_causal} questions): Ask about cause-effect chains spanning segments

4. IMPORTANT - Orthogonal Distributions (query_type and reasoning_type are SEPARATE fields):
   - Each question must have BOTH a query_type (multi_hop/structural/contextual) AND a reasoning_type (factual/relational/inferential/temporal/procedural/visual/causal)
   - These are TWO DIFFERENT fields - do NOT put reasoning types in the query_type field!
   - For example: A question can be query_type="multi_hop" with reasoning_type="procedural"
   - Ensure the final distribution matches both specified percentages

5. **IMPORTANT - Segment Identification**:
   - The content below contains segments formatted as "Segment N (HH:MM:SS - HH:MM:SS): text" or "Segment N [Doc: doc_id] (HH:MM:SS - HH:MM:SS): text" where N starts from 1
   - The "[Doc: doc_id]" tag indicates which document the segment belongs to (for multi-document bundles)
   - For each question-answer pair you generate, identify ALL segment numbers FROM which the question is derived
   - These segments are the source material that should be retrieved when someone asks this question
   - Record these segment numbers in the "segment_ids" field as a list of integers (e.g., [1, 4, 8])

6. For Each Question:
   - Must have complexity level {min_complexity} or higher
   - Generate the question FROM the identified segments (these segments are the source material)
   - Multi-hop questions must specify hop_count ({min_hops}-{max_hops})
   - Provide hop_contexts: a list where each hop includes "hop_number", "segment_ids" (the source segments for this hop), and "summary" (a concise summary describing the supporting segments).

7. Generate {num_pairs} distinct question and answer pairs.

The output should be a JSON object with a "pairs" field containing an array of {num_pairs} objects, where each object contains:
  - "question": the question, requiring understanding of the contexts/transcripts/segments without explicitly referencing the context/transcript/segments in the question
  - "answer": comprehensive answer from the contexts/transcripts/segments without explicitly referencing the context/transcript/segments in the answer
  - "question_complexity": numeric score {min_complexity}-5
  - "query_type": MUST be exactly one of these three values: "multi_hop", "structural", or "contextual" (NO other values allowed - do NOT use reasoning types here)
  - "reasoning_type": MUST be exactly one of these seven values: "factual", "relational", "inferential", "temporal", "procedural", "visual", or "causal" (this is DIFFERENT from query_type)
  - "segment_ids": list of segment numbers (e.g., [1, 4, 8]) that are the source material for this question (these should be retrieved when the question is asked)
  - "hop_count": number of hops ({min_hops}-{max_hops})
  - "hop_contexts": array of hop detail objects with "hop_number", "segment_ids", "summary"

CRITICAL: "query_type" and "reasoning_type" are TWO SEPARATE FIELDS with different allowed values. Do NOT mix them up:
  - query_type can ONLY be: "multi_hop", "structural", "contextual"
  - reasoning_type can ONLY be: "factual", "relational", "inferential", "temporal", "procedural", "visual", "causal"
"""

EVAL_SYSTEM_PROMPT = "You are an expert evaluator of question-answer pairs."

EVAL_PROMPT = """You are an expert evaluator of question-answer pairs.

You will evaluate multiple question-answer pairs from a document.

{qa_pairs_block}

<segments>
{segments_block}
</segments>

Evaluate EACH of the {num_pairs} QA pairs above.
Return a JSON object with a key "evaluations" that is a list of evaluations (one for each QA pair, in the same order as presented). Each evaluation must contain:
"relevance" (object with "score" 1-10 and "justification")
"accuracy" (object with "score" 1-10 and "justification")
"context_support" (object with "score" 1-10 and "justification")
"clarity" (object with "score" 1-10 and "justification")
"overall" (object with "score" float 1-10 and "assessment")
"improvements" (string)
"""

# ---------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# ---------------------------------------------------------------------------


def text_to_sentence_chunks(
    text: str, sentences_per_chunk: int = 5
) -> List[Dict[str, Any]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))

    chunks = []
    word_position = 0

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[i : i + sentences_per_chunk]
        chunk_text = ". ".join(chunk_sentences)
        if chunk_text and not chunk_text.endswith("."):
            chunk_text += "."

        chunk_words = chunk_text.split()
        chunks.append(
            {
                "text": chunk_text,
                "chunk_id": len(chunks) + 1,
                "word_count": len(chunk_words),
            }
        )

    return chunks


def dict_to_facts_block(artifacts: Dict[str, Any]) -> str:
    block = ""
    for category in [
        "key_concepts",
        "relationships",
        "themes",
        "entities",
        "processes",
        "insights",
        "technical_terms",
        "contextual_factors",
    ]:
        items = artifacts.get(category, [])
        if items:
            block += f"<{category}>\n"
            for item in items:
                text = item.get("text", "")
                desc = item.get("description", "")
                block += f"- {text}: {desc}\n"
            block += f"</{category}>\n\n"
    return block


def chunks_to_context_block(chunks: List[Dict[str, Any]]) -> str:
    # Mimicking "structured" sections from NVIDIA script
    lines = []
    lines.append("=== Section 1 ===")
    for chunk in chunks:
        lines.append(
            f"Segment {chunk['chunk_id']} (00:00:00 - 00:00:00): {chunk['text']}"
        )
    return "\n".join(lines)


def chunks_to_segments_block(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for chunk in chunks:
        lines.append(f"- Segment {chunk['chunk_id']}: {chunk['text']}")
    return "\n".join(lines)


def run_sdg_pipeline(input_dir: str, output_path: str, config_path: str):
    global_cfg = files.read_toml(config_path)
    llm = BaseLLM(LLMConfig(**global_cfg["llm"]))
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Iterate over markdown files
    for file_path in Path(input_dir).rglob("*.md"):
        print(f"Processing {file_path.name}...")
        text = file_path.read_text(encoding="utf-8")
        if len(text.strip()) == 0:
            continue

        chunks = text_to_sentence_chunks(text, sentences_per_chunk=5)
        if not chunks:
            continue

        print(f"  Generated {len(chunks)} chunks.")

        # 1. Extract Artifacts
        print("  Extracting artifacts...")
        artifacts_prompt = ARTIFACTS_PROMPT.format(text=text, max_artifacts=2)
        artifacts_res = llm.query(
            user_prompt=artifacts_prompt, system_prompt=ARTIFACTS_SYSTEM_PROMPT
        )
        if not isinstance(artifacts_res, dict):
            print(
                f"  Warning: Artifact extraction failed to return JSON dict. Skipping {file_path.name}."
            )
            continue

        # 2. Gen QA
        print("  Generating QA pairs...")
        facts_block = dict_to_facts_block(artifacts_res)
        context_block = chunks_to_context_block(chunks)

        qa_prompt = QA_GEN_PROMPT.format(
            facts_block=facts_block,
            context_block=context_block,
            query_counts_multi_hop=3,
            query_counts_structural=2,
            query_counts_contextual=2,
            reasoning_counts_factual=1,
            reasoning_counts_relational=1,
            reasoning_counts_inferential=1,
            reasoning_counts_temporal=1,
            reasoning_counts_procedural=1,
            reasoning_counts_visual=1,
            reasoning_counts_causal=1,
            min_hops=2,
            max_hops=3,
            min_complexity=3,
            num_pairs=5,
        )

        qa_res = llm.query(user_prompt=qa_prompt, system_prompt=QA_GEN_SYSTEM_PROMPT)
        if not isinstance(qa_res, dict) or "pairs" not in qa_res:
            print(
                f"  Warning: QA Generator failed to return pairs. Skipping {file_path.name}."
            )
            continue

        qa_pairs = qa_res["pairs"]

        # 3. QA Evaluations
        print("  Evaluating generated QA pairs...")
        qa_pairs_block = ""
        for i, pair in enumerate(qa_pairs):
            qa_pairs_block += f"=== QA Pair {i+1} ===\n\n"
            qa_pairs_block += f"QUESTION: {pair.get('question')}\n\n"
            qa_pairs_block += f"ANSWER: {pair.get('answer')}\n\n"
            qa_pairs_block += (
                f"CONTEXT (Relevant Segment IDs): {pair.get('segment_ids')}\n\n"
            )

        segments_block = chunks_to_segments_block(chunks)
        eval_prompt = EVAL_PROMPT.format(
            qa_pairs_block=qa_pairs_block,
            segments_block=segments_block,
            num_pairs=len(qa_pairs),
        )

        eval_res = llm.query(user_prompt=eval_prompt, system_prompt=EVAL_SYSTEM_PROMPT)
        if not isinstance(eval_res, dict) or "evaluations" not in eval_res:
            print(
                f"  Warning: Evaluator failed to return list. Output might be unverified."
            )
            evaluations = []
        else:
            evaluations = eval_res.get("evaluations", [])

        # Add everything to final dataset structure
        # (This mimics what the pipeline generates as flat DataFrame rows)
        results.append(
            {
                "file_name": str(file_path.name),
                "text": text,
                "chunks": chunks,
                "qa_pairs": qa_pairs,
                "qa_evaluations": evaluations,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"SDG complete. Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./example-config.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing markdowns"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/embed/stage0_sdg/generated_batch.json",
        help="Path for generated JSON",
    )
    args = parser.parse_args()

    run_sdg_pipeline(args.input_dir, args.output, args.config)
