import json
import numpy as np
from functools import wraps
from time import time
from typing import Union
from aisa.gen.providers import BaseInfo, ChatInfo, EmbedInfo
from aisa.utils import dictlist, logger


class ChatResponse:
    def single_query_wrapper(func: callable) -> callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start: float = time()
            input_text: str = kwargs.get("user_prompt", "")
            if input_text == "":
                input_text = args[0]
            prompt: str = kwargs.get("system_prompt", "")
            if prompt == "" and len(args) > 1:
                prompt = args[1]
            input_text = input_text + prompt
            output_text: str = ""
            verbose: bool = kwargs.get("verbose", True)
            info: ChatInfo = self.info

            # run and get timestamp
            result: Union[str, dict] = func(self, *args, **kwargs)
            elapsed_time: float = time() - start

            # format output
            if isinstance(result, dict):
                output_text = json.dumps(result)
            else:
                output_text = result

            # update model info (to calculate token usage and price)
            info.update_responses([input_text], [output_text], [elapsed_time])

            # logging
            if not verbose:
                return result

            log_model(info)
            log_cost(info, True)
            logger.opt(depth=1).log("TIME", "{time} sec", time=f"{elapsed_time:.2f}")
            logger.opt(depth=1).log("RESP", "{response}", response=output_text)
            return result

        return wrapper

    def chain_wrapper(func: callable) -> callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start: float = time()
            info: ChatInfo = self.info

            # read inputs
            input_texts: dictlist = kwargs.get("input_docs", [])
            if len(input_texts) == 0:
                input_texts = args[1]
            prompt: str = kwargs.get("prompt", "")
            if prompt == "":
                prompt = args[0]
            verbose: bool = kwargs.get("verbose", True)

            # format input
            if isinstance(input_texts[0], dict):
                new_texts: dictlist = [json.dumps(res) for res in input_texts]
            new_texts: dictlist = [prompt + ipt for ipt in new_texts]

            # run and get timestamp
            result: dictlist = await func(self, *args, **kwargs)
            elapsed_time: float = time() - start

            # format output
            if isinstance(result[0], dict):
                output_texts: dictlist = [json.dumps(res) for res in result]
            else:
                output_texts: dictlist = result

            # update model info (to calculate token usage and price)
            info.update_responses(new_texts, output_texts)

            # overwrite inputs
            if isinstance(input_texts[0], dict):
                input_texts: dictlist = [json.dumps(res) for res in input_texts]
            info.inputs.data[: len(input_texts)] = input_texts

            # logging
            if not verbose:
                return result

            log_model(info)
            log_cost(info, False)
            logger.opt(depth=1).log("TIME", "{time} sec", time=f"{elapsed_time:.2f}")
            logger.opt(depth=1).log("RESP", "...\n{last}", last=output_texts[-1])
            return result

        return wrapper


class EmbedResponse:
    def single_query_wrapper(func: callable) -> callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start: float = time()
            doc: str = kwargs.get("doc", "")
            if doc == "":
                doc = args[0]
            verbose: bool = kwargs.get("verbose", True)
            info: EmbedInfo = self.info

            # run and get timestamp
            result: np.ndarray = func(self, *args, **kwargs)
            elapsed_time: float = time() - start

            # update model info (to calculate token usage and price)
            info.update([doc], [elapsed_time])

            # logging
            if not verbose:
                return result

            log_model(info)
            log_cost(info, True)
            logger.opt(depth=1).log("TIME", "{time} sec", time=f"{elapsed_time:.2f}")
            logger.opt(depth=1).log("RESP", "{response}", response=result[:3])
            return result

        return wrapper

    def batch_wrapper(func: callable) -> callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start: float = time()
            docs: list[str] = kwargs.get("docs", [])
            if len(docs) == 0:
                docs = args[0]
            verbose: bool = kwargs.get("verbose", True)
            info: EmbedInfo = self.info

            # run and get timestamp
            result: np.ndarray = await func(self, *args, **kwargs)
            elapsed_time: float = time() - start

            # update model info (to calculate token usage and price)
            info.update(docs)

            # logging
            if not verbose:
                return result

            log_model(info)
            log_cost(info, False)
            logger.opt(depth=1).log("TIME", "{time} sec", time=f"{elapsed_time:.2f}")
            logger.opt(depth=1).log("RESP", "{response}", response=result[0][:3])
            return result

        return wrapper


def log_model(model_info: BaseInfo) -> None:
    logger.opt(depth=2, colors=True).log(
        "MODEL",
        "Model: <blue>{name}</blue> (<magenta>{provider}</magenta>)",
        name=model_info.name,
        provider=model_info.provider,
    )


def log_cost(model_info: ChatInfo, last_run: bool = True) -> None:
    if last_run:
        logger.opt(depth=2).log(
            "COST",
            "Input: ${cost} ({tokens} tok)",
            tokens=model_info.inputs.tokens[-1],
            cost=f"{model_info.inputs.cost[-1]:.6f}",
        )
        logger.opt(depth=2).log(
            "COST",
            "Outpt: ${cost} ({tokens} tok)",
            tokens=model_info.outputs.tokens[-1],
            cost=f"{model_info.outputs.cost[-1]:.6f}",
        )
    else:
        logger.opt(depth=2).log(
            "COST",
            "Input: ${cost} ({tokens} tok)",
            tokens=sum(model_info.inputs.tokens[: model_info.num_resp]),
            cost=f"{sum(model_info.inputs.cost[:model_info.num_resp]):.6f}",
        )
        logger.opt(depth=2).log(
            "COST",
            "Output: ${cost} ({tokens} tok)",
            tokens=sum(model_info.outputs.tokens[: model_info.num_resp]),
            cost=f"{sum(model_info.outputs.cost[: model_info.num_resp]):.6f}",
        )

    logger.opt(depth=2).log(
        "COST",
        "Total: ${cost} ({tokens} tok)",
        tokens=model_info.total_tokens,
        cost=f"{model_info.total_cost:.6f}",
    )
