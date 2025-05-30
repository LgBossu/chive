# I. IMPORTS

import re

# import signal
from multiprocessing import Process, Queue
from time import time

import chromadb
import torch
from loguru import logger
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from utils import load_paths as lp
from utils import log_setup  # noqa: F401

# class TimeoutException(Exception):
#     pass


# ###################################################################################

# II. SETUP

# II.1 Initialize the paths
logger.info("Initializing paths")
PATHS = lp.get_paths()
MODEL_PATH = PATHS["small_model_path"]
DATABASE_PATH = PATHS["chroma_db_path"]
MESSAGES_CATEGORIES_PATH = PATHS["messages_categories"]
BLACKLIST_PATH = PATHS["blacklist_categories"]


# II.2 Opening persistent database
logger.info("Accessing persistent database")
client = chromadb.PersistentClient(str(DATABASE_PATH))

conv_collection = client.get_collection("conversations")
mess_collection = client.get_collection("messages")


# II.3 Set up the LLM
logger.info("Setting up the LLM")

# Check if PyTorch is using the GPU
if torch.xpu.is_available():
    logger.success("PyTorch is using the GPU.")
else:
    logger.error("PyTorch is not using the GPU.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("xpu")


# ###################################################################################
# III. UTILS AND PROMPTS

# III.1 Prompts
logger.info("Setting up prompts")

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


# III.2 Utils
logger.info("Setting up utils")


def display_time(seconds: float, tz: int = 1, duration: bool = False) -> str:
    _, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    if not duration:
        hours = (hours + tz) % 24  # We are dealing with a date and account for timezone
    minutes, seconds = divmod(seconds, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    seconds = int(seconds)
    return f"{int(hours)}h {int(minutes)}m {seconds}s {int(milliseconds)}ms"


def tokenize_and_summon(
    message_content: str, tokenizer=tokenizer, model=model, max_output_length: int = 128
) -> str:
    full_prompt = prompt_begin + message_content + prompt_end
    logger.trace("Constructed full prompt")

    logger.trace("Tokenizing prompt")
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        # max_length=max_length,
        # truncation=True,
        # padding="max_length",
    ).to("xpu")

    logger.trace("Prompt tokenized. Generating output")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_output_length,
            min_length=1,  # Make sure it returns *something*
            repetition_penalty=1.1,
            # no_repeat_ngram_size=2,
            length_penalty=-0.1,  # Neutral length bias
            num_beams=5,  # Sampling + 1 beam = freeform
            early_stopping=True,
            temperature=None,
            do_sample=False,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    logger.trace("Output generated. Decoding.")

    raw_out = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.trace("Postprocessing output (pruning prompt from result)")
    postprocessed_out = raw_out[len(full_prompt) :]

    return postprocessed_out


def clean_llm_output(output):
    # Normalize case and spacing
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


def get_categories(message: str, max_output_length: int = 25) -> str:
    return clean_llm_output(tokenize_and_summon(message, max_output_length=max_output_length))


# III.3 Asynchronous handling to time out


# ASYNCHRONICITY WAS AI GENERATED

# VERSION 1 WITH SIGNALS

# def handler(signum, frame):
#     logger.info("Timeout reached. Raising exception.")
#     raise TimeoutException("Too long to execute. Timeout.")


# signal.signal(signal.SIGALRM, handler)


# def safe_get_categories(
#     message: str, message_id: str, max_output_length: int = 25, timeout: int = 30
# ) -> str:
#     signal.alarm(timeout)
#     try:
#         result = get_categories(message, max_output_length)
#         signal.alarm(0)
#         return result
#     except TimeoutException:
#         logger.warning(
#             f"Message has timed out after {timeout}s and will be added to blacklist"
#         )
#         with open(BLACKLIST_PATH, "a") as f:
#             f.write(f"{message_id}\n")
#         logger.info(f"Message {message_id} was blacklisted")
#         return None

# VERSION 2 WITH MULTIPROCESSING


def _generate_categories_worker(q: Queue, message, max_output_length):
    try:
        result = get_categories(message, max_output_length)
        q.put(result)
    except Exception as e:
        q.put(e)


def safe_get_categories(
    message: str, message_id: str, max_output_length: int = 25, timeout: int = 30
) -> str | None:
    q = Queue()
    p = Process(target=_generate_categories_worker, args=(q, message, max_output_length))
    p.start()
    p.join(timeout)

    if p.is_alive():
        logger.warning(
            f"Message {message_id} has timed out after {timeout}s and will be added to blacklist"  # noqa: E501
        )
        p.terminate()
        p.join()
        with open(BLACKLIST_PATH, "a") as f:
            f.write(f"{message_id}\n")
        logger.info(f"Message {message_id} was blacklisted")
        # raise TimeoutException(f"Message {message_id} exceeded timeout.")
        return None

    if q.empty():
        logger.error(f"Subprocess finished but returned nothing for message {message_id}.")
        return None

    result = q.get()

    if isinstance(result, Exception):
        logger.error(f"Error during category generation for message {message_id}: {result}")
        return None

    return result


# ###################################################################################
# IV. CATEGORIZATION

# IV.1 Load messages
logger.info("Loading messages from database")
raw_messages = mess_collection.get()
messages_ids = raw_messages["ids"]
messages_contents = raw_messages["documents"]

if messages_contents is None or messages_ids is None:
    logger.error("No messages found in the database (database.get() returned None elements).")
    raise ValueError("No messages found in the database.")

logger.success("Messages loaded successfully from the database")

logger.info("Loading previous categorizations")
try:
    with open(MESSAGES_CATEGORIES_PATH, "r") as f:
        categorized_messages = {line.split(",")[0] for line in f.read().splitlines()}
    logger.success("Previous categorizations loaded successfully.")
except FileNotFoundError as e:
    categorized_messages: set[str] = set()
    logger.warning("No previous categorizations found.")
    raise e


logger.info("Loading blacklist")
try:
    with open(BLACKLIST_PATH, "r") as f:
        blacklist = {line.strip() for line in f.read().splitlines()}
    logger.success("Blacklist loaded successfully.")
except FileNotFoundError as e:
    blacklist: set[str] = set()
    logger.warning("No blacklist found.")
    raise e


# IV.2 Categorize messages
logger.info("Categorizing messages")

starting_time = time()
last_checked_time = starting_time
processing_times = []
total_processed = 0
llm_processed = 0
llm_jettisoned = 0

logger.info(f"Starting categorization process at {display_time(starting_time)}")

for message_id, message_content in zip(messages_ids, messages_contents):
    message_start_time = time()
    total_processed += 1

    if message_id in blacklist:
        logger.debug(f"Message {message_id} is blacklisted. Skipping.")
        continue
    elif message_id in categorized_messages:
        logger.debug(f"Message {message_id} already categorized. Skipping.")
        continue
    else:
        llm_processed += 1
        logger.debug(f"Categorizing message {message_id}")
        categories = safe_get_categories(message_content, message_id)
        if categories is None:
            continue
        logger.debug(f"Categories for message {message_id}: {categories}")
        categorized_messages.add(message_id)
        with open(MESSAGES_CATEGORIES_PATH, "a") as f:
            f.write(f"{message_id},{categories}\n")
        logger.debug(f"Message {message_id} categorized successfully.")

        message_end_time = time()
        processing_time = message_end_time - message_start_time
        processing_times.append(processing_time)

    if message_end_time - last_checked_time > 60 * 3:
        logger.info(f"Since start: {total_processed} messages examined")
        logger.info(
            f"Processed {llm_processed} messages in {display_time(message_end_time - starting_time, duration=True)}"  # noqa: E501
        )
        logger.info(
            f"Average processing time: {sum(processing_times) / len(processing_times) if processing_time else 0:.2f} seconds"  # noqa: E501
        )
        last_checked_time = message_end_time
