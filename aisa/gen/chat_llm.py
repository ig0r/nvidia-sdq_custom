from typing import Union
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    AIMessage,
)
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from aisa.utils import dictlist, files
from aisa.gen.decorators import ChatResponse
from aisa.gen.prompts import clean_json
from aisa.gen.providers import Provider, ChatInfo, CHAT_MODELS
import aisa.gen.ollama_api as ollama_api


class LLMConfig(BaseModel):
    json_mode: bool = Field(alias="json_mode", default=False)
    model: str = Field(alias="model", default="gemini-2.0-flash")
    request_timeout: int = Field(alias="request_timeout", default=30)
    temperature: float = Field(alias="temperature", default=0.0)
    max_tokens: int = Field(alias="max_tokens", default=4000)
    max_input_tokens: int = Field(alias="max_input_tokens", default=3000)
    max_chain_tokens: int = Field(alias="max_chain_tokens", default=30000)
    chunk_size: int = Field(alias="chunk_size", default=20)
    prompt_lib: str = Field(alias="prompt_lib", default="")

    @property
    def name(self) -> str:
        clean_name: str = files.clean_filename(self.model)
        temp_str: str = str(self.temperature).replace(".", "")
        return clean_name + f"_t{temp_str}"


def _init_openai_model(cfg: LLMConfig, api_key: str) -> BaseChatModel:
    model = ChatOpenAI(
        api_key=api_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    if cfg.json_mode:
        model.model_kwargs["response_format"] = {"type": "json_object"}
    return model


def _init_google_model(cfg: LLMConfig, api_key: str) -> BaseChatModel:
    model = ChatGoogleGenerativeAI(
        model=cfg.model,
        google_api_key=api_key,
        max_output_tokens=cfg.max_tokens,
    )
    return model


def _init_ollama_model(cfg: LLMConfig, api_key: str = "") -> BaseChatModel:
    ollama_api.check_existing_model(cfg.model)
    try:
        model = ChatOllama(
            model=cfg.model,
            temperature=cfg.temperature,
            num_predict=cfg.max_tokens,
            format="json" if cfg.json_mode else None,
        )
    except Exception as e:
        raise ValueError(f"Error initializing model `{cfg.model}`: {e}")

    return model


CHATMODEL_INIT: dict[Provider, callable] = {
    Provider.OPENAI: _init_openai_model,
    Provider.GOOGLE: _init_google_model,
    Provider.OLLAMA: _init_ollama_model,
}


class BaseLLM:
    def __init__(
        self,
        llm_cfg: LLMConfig = LLMConfig(),
        out_dir: str = "./data/test",
    ) -> None:
        info: ChatInfo = CHAT_MODELS.get(llm_cfg.model, ChatInfo())
        self.cfg: LLMConfig = llm_cfg
        self.info: ChatInfo = info
        self.info.name = self.cfg.model
        self.model: BaseChatModel = CHATMODEL_INIT[self.info.provider](
            llm_cfg, info.api_key
        )
        self.base_out: str = files.os_path(out_dir + f"/{self.cfg.name}")

    @ChatResponse.single_query_wrapper
    def query(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful AI assistant",
        verbose: bool = True,
    ) -> Union[str, dict]:
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response: AIMessage = self.model.invoke(messages)
        resp_text: str = response.content
        return clean_json(resp_text) if self.cfg.json_mode else str(resp_text)

    def read_prompt(self, prompt_name: str) -> str:
        prompt: str = ""
        prompt_path: str = files.os_path(self.cfg.prompt_lib + f"/{prompt_name}.txt")
        if files.exists(prompt_path):
            prompt = files.read_text_file(prompt_path)
            return prompt
        raise FileNotFoundError(f"Prompt '{prompt_name}' does not exist")

    @ChatResponse.chain_wrapper
    async def run_chain(
        self,
        prompt: str,
        input_docs: dictlist,
        verbose: bool = True,
    ) -> dictlist:
        main_prompt: PromptTemplate = PromptTemplate(
            template=prompt,
            input_variables=list(input_docs[0].keys()),
        )
        chain: RunnableSequence = main_prompt | self.model
        results: list[AIMessage] = await chain.abatch(input_docs)
        if self.cfg.json_mode:
            return [clean_json(result.content.strip()) for result in results]
        return [result.content for result in results]
