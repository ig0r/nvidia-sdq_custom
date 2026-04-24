from abc import ABC, abstractmethod
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from aisa.utils import files
from aisa.gen.providers import Provider, EmbedInfo, EMBED_MODELS
from aisa.gen.decorators import EmbedResponse
import aisa.gen.ollama_api as ollama_api


class EmbedConfig(BaseModel):
    model: str = Field(alias="model", default="sentence-transformers/all-MiniLM-L6-v2")
    path: str = Field(alias="path", default="")
    embed_dim: int = Field(alias="embed_dim", default=384)
    chunk_size: int = Field(alias="chunk_size", default=256)
    recursive_overlap: int = Field(alias="recursive_overlap", default=50)
    prefix: str = Field(alias="prefix", default="")

    @property
    def name(self) -> str:
        return files.clean_filename(self.model)


EMBEDMODEL_INIT: dict[Provider, callable] = {
    Provider.OPENAI: lambda model_name, key: OpenAIEmbeddings(
        api_key=key,
        model=model_name,
    ),
    Provider.GOOGLE: lambda model_name, key: GoogleGenerativeAIEmbeddings(
        api_key=key,
        model=model_name,
    ),
    Provider.HUGGINGFACE: lambda model_name, key: HuggingFaceEmbeddings(
        model_name=key if model_name == "custom" else model_name,
        model_kwargs={"trust_remote_code": True},
    ),
    Provider.OLLAMA: lambda model_name, key: OllamaEmbeddings(
        model=model_name,
    ),
}


class Embedder:
    def __init__(
        self,
        embed_cfg: EmbedConfig = EmbedConfig(),
        out_dir: str = "./data/test",
    ):
        ollama_models: list[str] = ollama_api.list_models()
        info: EmbedInfo = None
        self.cfg: EmbedConfig = embed_cfg
        if self.cfg.model in EMBED_MODELS:
            info = EMBED_MODELS[self.cfg.model]
        elif self.cfg.model in ollama_models:
            info = EmbedInfo(Provider.OLLAMA)
        else:
            info = EmbedInfo(Provider.HUGGINGFACE)
        self.info: EmbedInfo = info
        self.info.api_key = self.cfg.path if self.cfg.model == "custom" else None
        self.info.name = self.cfg.model
        self.model: Embeddings = EMBEDMODEL_INIT[self.info.provider](
            info.name, info.api_key
        )
        self.base_out: str = files.os_path(out_dir + f"/{self.cfg.name}")

    @EmbedResponse.single_query_wrapper
    def embed_doc(self, doc: str, verbose: bool = True) -> np.ndarray:
        embed: list[float] = self.model.embed_query(doc)
        self.cfg.embed_dim = len(embed)
        self.info.embed_dim = self.cfg.embed_dim
        return np.array(embed)

    def non_async_embed_docs(self, docs: list[str]) -> np.ndarray:
        embeds: list[list[float]] = self.model.embed_documents(docs)
        self.cfg.embed_dim = len(embeds[0]) if embeds else 0
        self.info.embed_dim = self.cfg.embed_dim
        return np.array(embeds)

    @EmbedResponse.batch_wrapper
    async def embed_docs(self, docs: list[str], verbose: bool = True) -> np.ndarray:
        embeds: list[list[float]] = await self.model.aembed_documents(docs)
        self.cfg.embed_dim = len(embeds[0]) if embeds else 0
        self.info.embed_dim = self.cfg.embed_dim
        return np.array(embeds)

    def get_token_count(self, doc: str) -> np.ndarray:
        pass

    def recursive_split(self, doc: str) -> list[str]:
        pass

    def partition_inputs(self, docs: list[str], chunk_size: int = 100) -> np.ndarray:
        chunks = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]
        embed_batch: np.ndarray = None
        for chunk in chunks:
            embeddings = self.non_async_embed_docs(chunk)
            if embed_batch is None:
                embed_batch = embeddings
            else:
                embed_batch = np.concatenate([embed_batch, embeddings], axis=0)

        return embed_batch


# NOTE: DEPRECATED
class HFEmbedder(Embedder):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model = AutoModel.from_pretrained(self.cfg.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        self.embed_dim = self.tokenizer.model_max_length

    def embed_doc(self, doc: str) -> np.ndarray:
        encoded_input = self.tokenizer(
            doc, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = _mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        return F.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()

    def embed_docs(self, docs: list[str]) -> np.ndarray:
        encoded_input = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.embed_dim,
        )
        self.model.eval()
        with torch.no_grad():
            encoded_input = {
                k: v.to(self.model.device) for k, v in encoded_input.items()
            }
            model_output = self.model(**encoded_input)

        sentence_embeddings = _mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        return F.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()

    def get_token_count(self, doc: str) -> int:
        encoded_input = self.tokenizer(
            doc, padding=False, truncation=False, return_tensors="pt"
        )
        return len(encoded_input["input_ids"][0])

    def recursive_split(
        self, doc: str, size: int = 256, overlap: int = 50
    ) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=self.get_token_count,
        )
        return text_splitter.split_text(doc)


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
