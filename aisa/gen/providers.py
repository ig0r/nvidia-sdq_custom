import os
from enum import Enum
import tiktoken as tikt
from dotenv import load_dotenv

load_dotenv()
ENCODING: tikt.Encoding = tikt.encoding_for_model("gpt-3.5-turbo")


class Provider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


API_KEYS: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.GOOGLE: "GOOGLE_API_KEY",
    Provider.OLLAMA: "",
    Provider.HUGGINGFACE: "",
}


class BaseInfo:
    def __init__(self, provider: Provider = Provider.OLLAMA):
        self.name: str = ""
        self.provider: Provider = provider
        self.api_key_env: str = API_KEYS[provider]
        self.api_key: str = os.getenv(self.api_key_env)
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.embed_dim: int = 0
        self.num_resp: int = 0

        if (
            provider != Provider.OLLAMA
            and provider != Provider.HUGGINGFACE
            and not self.api_key
        ):
            raise ValueError(f"Missing {self.api_key_env} in env vars")


class ChatIO:
    def __init__(self, price: float = 0.0):
        self.price: float = price  # per 1M token
        self.tokens: list[int] = []
        self.cost: list[float] = []
        self.data: list[str] = []

    def add_values(self, text: str) -> None:
        self.data.append(text)
        self.tokens.append(len(ENCODING.encode(self.data[-1])))
        self.cost.append((self.tokens[-1] / 1e6) * self.price)


class ChatInfo(BaseInfo):
    def __init__(
        self,
        provider: Provider = Provider.OLLAMA,
        input_price: float = 0.00,
        output_price: float = 0.00,
    ):
        super().__init__(provider)
        self.inputs: ChatIO = ChatIO(input_price)
        self.outputs: ChatIO = ChatIO(output_price)
        self.times: list[float] = []

    def update_responses(
        self,
        input_texts: list[str],
        output_texts: list[str],
        times: list[float] = [],
    ) -> None:
        [self.inputs.add_values(text) for text in input_texts]
        [self.outputs.add_values(text) for text in output_texts]
        self.total_tokens = sum(self.inputs.tokens) + sum(self.outputs.tokens)
        self.total_cost = sum(self.inputs.cost) + sum(self.outputs.cost)
        self.num_resp = len(output_texts)

        if len(times) > 0:
            self.times.extend(times)
            return

        self.times.extend([-1.00 for _ in range(len(input_texts))])


class EmbedInfo(BaseInfo):
    def __init__(
        self,
        provider: Provider = Provider.HUGGINGFACE,
        price: float = 0.00,
        dim: int = 0,
    ):
        super().__init__(provider)
        self.inputs: ChatIO = ChatIO(price)
        self.outputs: ChatIO = ChatIO(0.00)
        self.price: float = price
        self.embed_dim: int = dim

    def update(self, input_texts: list[str], times: list[float] = []) -> None:
        [self.inputs.add_values(text) for text in input_texts]
        [self.outputs.add_values("") for _ in input_texts]
        self.total_tokens = sum(len(ENCODING.encode(text)) for text in input_texts)
        self.total_cost = (self.total_tokens / 1e6) * self.price
        self.num_resp = len(input_texts)


CHAT_MODELS: dict[str, ChatInfo] = {
    "gpt-4o-mini": ChatInfo(Provider.OPENAI, 0.15, 0.60),
    "gemini-2.0-flash": ChatInfo(Provider.GOOGLE, 0.10, 0.40),
}

EMBED_MODELS: dict[str, EmbedInfo] = {
    "text-embedding-3-small": EmbedInfo(Provider.OPENAI, 0.02, 1536),
    "gemini-embedding-001": EmbedInfo(Provider.GOOGLE, 0.15, 3072),
}
