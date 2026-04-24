import requests
import subprocess
import ollama
from ollama import ListResponse, ProcessResponse
from aisa.utils.log import logger, heading
from aisa.utils.helpers import byte_to_gb

MAIN_ENDPOINT = "http://localhost:11434"


def list_models(verbose: bool = False) -> list[str]:
    installed_models: list[str] = []
    try:
        model_list: ListResponse = ollama.list()
        for model in model_list.models:
            installed_models.append(model.model)

        if verbose:
            heading("Installed Ollama Models")
            logger.info("Name | Size | Parameters | Quantization")
            for model in model_list.models:
                logger.log(
                    "MODEL",
                    f"{model.model} | "
                    + f"{byte_to_gb(model.size):.2f} GB | "
                    + f"{model.details.parameter_size} | "
                    + f"{model.details.quantization_level}",
                )
    except requests.exceptions.ConnectionError:
        logger.error("Error: Cannot connect to Ollama. Make sure Ollama is running.")
        exit()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        exit()

    return installed_models


def check_existing_model(model_name: str) -> None:
    installed_models: list[str] = list_models()
    if model_name not in installed_models:
        logger.error(f"Model '{model_name}' is not installed.")
        logger.info(f"Please install the model using: `ollama install {model_name}`")
        exit()
    return


def kill_running_models() -> None:
    running_models: list[str] = []
    heading("Killing Running Ollama Processes")
    try:
        model_list: ProcessResponse = ollama.ps()

        for model in model_list.models:
            running_models.append(model.model)
    except requests.exceptions.ConnectionError:
        logger.error("Error: Cannot connect to Ollama. Make sure Ollama is running.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

    finally:
        if running_models:
            logger.info("Name | VRAM")
            for model in model_list.models:
                logger.log(
                    "MODEL", f"{model.model} | {byte_to_gb(model.size_vram):.2f} GB"
                )
                subprocess.run(["ollama", "stop", model.model])
        else:
            logger.info("No running models found.")
