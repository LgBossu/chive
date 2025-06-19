# from time import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from backend.utils.path_utils import get_paths

# def display_time(seconds: float, tz: int = 1, duration: bool = False) -> str:
#     _, seconds = divmod(seconds, 86400)
#     hours, seconds = divmod(seconds, 3600)
#     if not duration:
#         hours = (hours + tz) % 24  # We are dealing with a date and account for timezone
#     minutes, seconds = divmod(seconds, 60)
#     milliseconds = (seconds - int(seconds)) * 1000
#     seconds = int(seconds)
#     return f"{int(hours)}h {int(minutes)}m {seconds}s {int(milliseconds)}ms"


class CategorizerModel(ABC):
    """
    An abstract base class for categorizer models.

    The role of these models is to categorize text messages dynamically.
    """

    @abstractmethod
    def _check_model_path(self, model_path: Optional[Path]) -> Path:
        """
        Checks if the provided model path is valid and returns it.
        If no path is provided, a default path is used.

        :param model_path: Optional path to the model file.
        :return: A valid model path.
        :raises FileNotFoundError: If the model file does not exist.
        :raises ValueError: If the model path is not a directory.
        """
        pass

    @abstractmethod
    def _check_hardware_acceleration(self) -> str:
        """
        Checks if the hardware acceleration is available.
        Raises an error if it is not available.

        :raises RuntimeError: If hardware acceleration is not available.
        :return: The type of hardware acceleration available (e.g., "cuda", "xpu"...)
        :rtype: str
        """
        pass

    @abstractmethod
    def _load_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """
        Loads the model and tokenizer from the specified path.

        In implemenations, do not forget setting the device map to `self.hardware`
        to ensure the model is loaded on the correct device.

        :return: A tuple containing the tokenizer and the model.
        :rtype: Tuple[PreTrainedTokenizer, PreTrainedModel]
        """
        pass

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initializes the categorizer model, checking the model path and hardware acceleration,
        before loading the model and tokenizer.

        :param model_path: Optional path to the model file.
        """  # noqa: E501

        self.model_path = self._check_model_path(model_path)
        self.hardware = self._check_hardware_acceleration()
        self.tokenizer, self.model = self._load_model()

    @property
    @abstractmethod
    def prompt_parts(self) -> Tuple[str, str]:
        """
        Gets the parts of the prompt used for categorization,
        in a general expected structure of (prefix, suffix), meant to be used as :
        prefix + [message to categorize] + suffix

        It is important to tune the prompts to the specific model being used.
        This method should be implemented in subclasses to provide the specific prompt parts.


        :return: A tuple containing the prefix and suffix of the prompt.
        :rtype: Tuple[str, str]
        """
        # TODO : also consider storing the prompts in a separate proper file,
        # to avoid hardcoding and allow for easy editing and tuning.
        pass

    def _construct_full_prompt(self, message: str) -> str:
        """
        Constructs the full prompt by combining the prefix, the message to categorize,
        and the suffix.

        :param message: The message to categorize.
        :return: The full prompt string.
        :rtype: str
        """
        logger.debug("Constructing full prompt for categorization.")
        prefix, suffix = self.prompt_parts
        return prefix + message + suffix

    def _tokenize(self, full_prompt: str):
        """
        Tokenizes the full prompt using the tokenizer.

        :param full_prompt: The full prompt string to tokenize.
        :return: Tokenized input tokens ready for the model.
        """
        logger.debug("Tokenizing the full prompt.")
        input_tokens = self.tokenizer(
            full_prompt,
            return_tensors="pt",
        ).to(self.hardware)
        return input_tokens

    @property
    @abstractmethod
    def model_parameters(self) -> Dict[str, Any]:
        """
        Returns the parameters of the model used for categorization.
        This should be implemented in subclasses to provide specific model parameters.

        :return: A dictionary containing the model parameters.
        :rtype: Dict[str, Any]
        """
        # TODO : also consider storing the model parameters in a separate proper file,
        # to avoid hardcoding and allow for easy editing and tuning.
        pass

    def _generate(self, input_tokens, max_output_length: int = 128):
        """
        Generates a response from the model based on the input tokens.

        :param input_tokens: The tokenized input ready for the model.
        :param max_output_length: The maximum length of the generated output.
        :return: The generated output from the model.
        """
        # TODO : max_output length should be configurable, not hardcoded.

        logger.debug("Generating response from the model.")
        with torch.no_grad():
            output = self.model.generate(
                **input_tokens,
                max_new_tokens=max_output_length,
                **self.model_parameters,
            )
        return output

    def _decode_and_extract(self, full_prompt: str, output) -> str:
        """
        Decodes the generated output and extracts the relevant part.

        :param full_prompt: The full prompt used for categorization.
        :param output: The generated output from the model.
        :return: The extracted response from the model.
        """
        logger.debug("Decoding and extracting response from the model output.")
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract the part after the full prompt
        response = decoded_output[len(full_prompt) :].strip()
        return response

    @abstractmethod
    def _clean_llm_output(self, output: str) -> str:
        """
        Cleans the LLM output to ensure it is in a usable format.

        :param output: The raw output from the model.
        :return: The cleaned output string.
        """
        # TODO : implement this method in subclasses to provide specific cleaning logic.
        pass

    @abstractmethod
    def _parse_categories(self, output: str) -> List[str]:
        """
        Parses the cleaned output to extract categories.

        :param output: The cleaned output string.
        :return: A list of categories extracted from the output.
        """
        # TODO : implement this method in subclasses to provide specific parsing logic.
        pass

    def categorize(self, message: str) -> List[str]:
        """
        Categorizes the given message by constructing a prompt, tokenizing it,
        generating a response from the model, decoding the output, cleaning it,
        and finally parsing the categories.

        :param message: The message to categorize.
        :return: A list of categories for the message.
        """
        logger.info(f"Categorizing message: {message}")
        full_prompt = self._construct_full_prompt(message)
        input_tokens = self._tokenize(full_prompt)
        output = self._generate(input_tokens)
        decoded_output = self._decode_and_extract(full_prompt, output)
        cleaned_output = self._clean_llm_output(decoded_output)
        categories = self._parse_categories(cleaned_output)

        logger.info(f"Categories found: {categories}")
        return categories
