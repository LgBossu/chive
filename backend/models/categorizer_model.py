# TODO : update docstrings
import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from backend.utils.path_utils import get_paths


class CategorizerModel(ABC):
    """
    An abstract base class for torch-based categorizer models.

    The role of these models is to categorize text messages dynamically.

    ABSTRACT METHODS:
    - _check_model_path: Checks if the provided model path is valid and returns it.
    - _check_hardware_acceleration: Checks if the hardware acceleration is available.
    - _load_model: Loads the model and tokenizer from the specified path.
    - prompt_parts: Gets the parts of the prompt used for categorization.
    - model_parameters: Returns the parameters of the model used for categorization.
    - _clean_llm_output: Cleans the LLM output to ensure it is in a usable format.
    - _parse_categories: Parses the cleaned output to extract categories.

    METHODS:
    - __init__: Initializes the categorizer model, checking the model path and hardware acceleration,
      before loading the model and tokenizer.
    - _construct_full_prompt: Constructs the full prompt by combining the prefix, the message to categorize,
      and the suffix.
    - _tokenize: Tokenizes the full prompt using the tokenizer.
    - _generate: Generates a response from the model based on the input tokens.
    - _decode_and_extract: Decodes the generated output and extracts the relevant part.
    - categorize: Categorizes the given message by constructing a prompt, tokenizing it,
      generating a response from the model, decoding the output, cleaning it,
      and finally parsing the categories.
    """  # noqa: E501

    class ExceedingSafetyLimitError(Exception):
        """
        Exception raised when the input length exceeds the safety limit.
        This is used to prevent unexpected behavior in the categorization process.
        """

        pass

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

        :param model_path: Optional path to the model file. If not provided, a default path is used.

        Initializes attributes:
        - model_path: The path to the model file.
        - hardware: The type of hardware acceleration available (e.g., "cuda", "xpu"...)
        - tokenizer: The tokenizer for the model.
        - model: The model for categorization.
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

    @property
    @abstractmethod
    def safe_max_length(self) -> int:
        """
        Returns the maximum length (in tokens) that the model can handle reliably.
        This should be implemented in subclasses to provide specific maximum length values.

        :return: The maximum length in tokens.
        :rtype: int
        """
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

    def categorize(
        self,
        message: str,
        max_output_length: int = 128,
        safety_input_length: int = 2048,
    ) -> List[str]:
        """
        Categorizes the given message by constructing a prompt, tokenizing it,
        generating a response from the model, decoding the output, cleaning it,
        and finally parsing the categories.

        :param message: The message to categorize.
        :return: A list of categories for the message.
        """
        logger.trace("Categorizing message")
        full_prompt = self._construct_full_prompt(message)
        input_tokens = self._tokenize(full_prompt)
        if input_tokens.input_ids.shape[1] > safety_input_length:
            logger.warning(
                f"Input length {input_tokens.input_ids.shape[1]} exceeds safety limit of {safety_input_length} tokens. "  # noqa: E501
            )
            raise self.ExceedingSafetyLimitError("Input length exceeds safety limit. ")
        output = self._generate(input_tokens, max_output_length=max_output_length)
        decoded_output = self._decode_and_extract(full_prompt, output)
        cleaned_output = self._clean_llm_output(decoded_output)
        categories = self._parse_categories(cleaned_output)

        logger.trace(f"Categories found: {categories}")
        return categories

    @abstractmethod
    def __del__(self):
        """
        Cleans up the model and tokenizer when the instance is deleted.
        This is important to free up resources, especially for large models.

        The method should be overridden in subclasses to ensure proper cleanup of the cache.
        """
        logger.debug("Cleaning up the categorizer model and tokenizer.")
        del self.tokenizer
        del self.model

    @property
    @abstractmethod
    def timeout(self) -> int:
        """
        Returns the timeout for the categorization process.
        This should be implemented in subclasses to provide specific timeout values.

        :return: The timeout value in seconds.
        :rtype: int
        """
        pass

    @property
    @abstractmethod
    def recommended_output_length(self) -> int:
        """
        Returns the recommended output length for the categorization process.
        This is used to limit the length of the generated output.

        :return: The recommended output length in tokens.
        :rtype: int
        """
        pass


class Categorizer0(CategorizerModel):
    """
    This default categorizer model uses Llama 3.2, and plugs into 'xpu' hardware acceleration
    to support Intel GPU calculations.
    """

    def _check_model_path(self, model_path: Optional[Path]) -> Path:
        """
        Checks if the provided model path is valid and returns it.
        If no path is provided, a default path is used.

        :param model_path: Optional path to the model file.
        :return: A valid model path.
        :raises FileNotFoundError: If the model file does not exist.
        :raises ValueError: If the model path is not a directory.
        """
        if model_path is None:
            model_path = get_paths().small_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not model_path.is_dir():
            raise ValueError(f"Model path {model_path} is not a directory.")
        return model_path

    def _check_hardware_acceleration(self) -> str:
        """
        Checks if the hardware acceleration is available.
        Raises an error if it is not available.

        :raises RuntimeError: If hardware acceleration is not available.
        :return: The type of hardware acceleration available (e.g., "cuda", "xpu"...)
        :rtype: str
        """
        if torch.xpu.is_available():
            logger.success("Using Intel GPU (XPU) for hardware acceleration.")
            return "xpu"  # TODO : should xpu be hardcoded here?
            # Probably not, but it is the only one we support for now.
        else:
            logger.error("Intel GPU (XPU) is not available. Please check your setup.")
            raise RuntimeError("Intel GPU (XPU) is not available. Please check your setup.")

    def _load_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """
        Loads the model and tokenizer from the specified path.

        The model is mapped to the hardware device specified by `self.hardware`.

        :return: A tuple containing the tokenizer and the model.
        :rtype: Tuple[PreTrainedTokenizer, PreTrainedModel]
        """
        logger.info(f"Loading model from {self.model_path} on [{self.hardware}] device.")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.hardware)
        return tokenizer, model

    def __init__(self, model_path: Path | None = None):
        super().__init__(model_path)
        mem_bytes = (
            torch.xpu.memory_allocated()
        )  # Remember that this class implements XPU hardware acceleration.
        mem_gb = mem_bytes / (1024**3)
        logger.info(f"Memory info : {mem_gb:.3f} GB allocated on GPU.")

    @property
    def prompt_parts(self) -> Tuple[str, str]:
        """
        Gets the parts of the prompt used for categorization,
        in a general expected structure of (prefix, suffix), meant to be used as :
        prefix + [message to categorize] + suffix

        :return: A tuple containing the prefix and suffix of the prompt.
        :rtype: Tuple[str, str]
        """

        prompt_begin = """You are a helpful assistant. Given a user message, your job is to identify its main topic(s) or emotional theme(s) in a few simple words.

- Return a short, comma-separated list of themes.
- Output only the list.
- If the message has no meaningful content, respond with: none.
- End your response with <END>.

Here are some examples:

MESSAGE: [ok lol!]
CATEGORIZATION: none <END>

MESSAGE: [I'm feeling a bit overwhelmed, but also proud of the work I did today.]
CATEGORIZATION: stress, accomplishment, self-reflection <END>

MESSAGE: [I just made saffron rice with lemon and it actually turned out amazing!]
CATEGORIZATION: cooking, food, pride <END>

MESSAGE: ["""  # noqa: E501

        prompt_end = """]
CATEGORIZATION:"""

        return prompt_begin, prompt_end

    @property
    def model_parameters(self) -> Dict[str, Any]:
        """
        Returns the parameters of the model used for categorization.
        This should be implemented in subclasses to provide specific model parameters.

        :return: A dictionary containing the model parameters.
        :rtype: Dict[str, Any]
        """
        parameters = {
            "min_length": 1,  # Make sure it returns *something*
            "repetition_penalty": 1.1,
            # no_repeat_ngram_size:2,
            "length_penalty": -0.1,  # Neutral length bias
            "num_beams": 5,  # Sampling + 1 beam = freeform
            "early_stopping": True,
            "temperature": None,
            "do_sample": False,
            "top_p": None,
            "top_k": None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        return parameters

    @property
    def safe_max_length(self) -> int:
        """
        Returns the maximum length (in tokens) that the model can handle reliably.
        This should be implemented in subclasses to provide specific maximum length values.

        :return: The maximum length in tokens.
        :rtype: int
        """
        return 2048

    def _clean_llm_output(self, output: str) -> str:
        """
        Cleans the LLM output to ensure it is in a usable format.

        :param output: The raw output from the model.
        :return: The cleaned output string.
        """
        logger.trace("Cleaning output")
        output = output.lower().strip()

        # Try to split on a variant of <end>
        end_match = re.search(r"<.*end.*>", output)
        if end_match:
            output = output[: end_match.start()].strip()
        else:
            arbitrary_match = re.search(r"<.*>", output)
            if arbitrary_match:
                output = output[: arbitrary_match.start()].strip()
        logger.trace(f"Output after <end> removal: {output}")

        # Remove brackets and quotes
        output = output.replace("[", "").replace("]", "")
        output = output.replace('"', "").replace("'", "")

        # Remove unwanted prefixes (hallucinated headers)
        for token in ["categorization:", "category:", "# output", "response:"]:
            output = output.replace(token, "")

        # Collapse extra whitespace
        output = re.sub(r"\s+", " ", output).strip()

        # Default to 'none' if output is empty
        if not output or output in ["<end>", "none <end>"]:
            output = "none"

        output = output.replace(", ", ";")

        logger.trace(f"Final cleaned output: {output}")

        return output

    def _parse_categories(self, output: str) -> List[str]:
        """
        Parses the cleaned output to extract categories.

        :param output: The cleaned output string.
        :return: A list of categories extracted from the output.
        """
        logger.trace("Parsing categories from output")
        # Split by semicolon and strip whitespace
        categories = [cat.strip() for cat in output.split(";") if cat.strip()]
        logger.trace(f"Parsed categories: {categories}")
        return categories

    def __del__(self):
        """
        Cleans up the model and tokenizer when the instance is deleted.
        This is important to free up resources, especially for large models.

        The method should be overridden in subclasses to ensure proper cleanup of the cache.
        """
        super().__del__()
        torch.xpu.empty_cache()
        logger.debug("Categorizer0 model and tokenizer cleaned up.")

    @property
    def timeout(self) -> int:
        """
        Returns the timeout for the categorization process.
        This should be implemented in subclasses to provide specific timeout values.

        :return: The timeout value in seconds.
        :rtype: int
        """
        return 30

    @property
    def recommended_output_length(self) -> int:
        """
        Returns the recommended output length for the categorization process.
        This is used to limit the length of the generated output.

        :return: The recommended output length in tokens.
        :rtype: int
        """
        return 30


class AvailableCategorizers(Enum):
    """
    Enum for available categorizer models.
    """

    CATEGORIZER_0 = Categorizer0
