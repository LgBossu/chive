# I. IMPORTS

import datetime
import re

# import signal
from multiprocessing import Process, set_start_method
from time import sleep

import chromadb
from loguru import logger
from utils import load_paths as lp

set_start_method("spawn", force=True)


# ###################################################################################


# IV.2 Categorize messages
def categorization_loop(
    MODEL_PATH: str,
    MESSAGES_CATEGORIES_PATH: str,
    LOG_FILE: str,
    BLACKLIST_PATH: str,
    prompt_begin: str,
    prompt_end: str,
    messages_ids: list[str],
    messages_contents: list[str],
    categorized_messages: set[str],
    blacklist: set[str],
):
    import gc
    import re  # noqa: F811
    import sys
    from time import time  # noqa: F811

    import torch
    from loguru import logger
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    SAFE_MAX_LENGTH = 2048

    # logger.info("Setting up the logger")
    # Set up the logger
    logger.remove()
    logger.add(
        sink=sys.stdout,  # Output to the console
        format="<level>{level:<10} | {message}</>",
        level="INFO",
        colorize=True,
    )
    logger.add(
        sink=LOG_FILE,  # Output to the log file
        format="{time} | {level:<10} | {name}:{function}:{line} - {message}",
        level="TRACE",
        backtrace=True,
        diagnose=True,
        colorize=True,
    )
    logger.info("Subprocess logger set up")
    logger.info(f"Set maximum message length to {SAFE_MAX_LENGTH} tokens")

    if torch.xpu.is_available():
        logger.success("PyTorch is using the GPU.")
    else:
        logger.error("PyTorch is not using the GPU.")
        raise SystemError("PyTorch is not using the GPU.")

    logger.info(
        f"Memory info before loading model: {torch.xpu.memory_allocated()} / {torch.xpu.memory_reserved()}"  # noqa: E501
    )  # noqa: E501

    logger.info("Instantiating LLM model ")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("xpu")
    logger.info("Instantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    logger.info(
        f"Memory info before clearing cache : {torch.xpu.memory_allocated()} / {torch.xpu.memory_reserved()}"  # noqa: E501
    )
    logger.info("Clearing cache and collecting garbage")
    gc.collect()
    torch.xpu.empty_cache()
    logger.info(f"Memory info : {torch.xpu.memory_allocated()} / {torch.xpu.memory_reserved()}")  # noqa: E501

    logger.info("Setting up utils")

    def display_time(seconds: float, tz: int = 1, duration: bool = False) -> str:
        if not duration:
            _, seconds = divmod(seconds, 86400)
            hours, seconds = divmod(seconds, 3600)
            hours = (hours + tz) % 24  # We are dealing with a date and account for timezone
        else:
            hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        milliseconds = (seconds - int(seconds)) * 1000
        seconds = int(seconds)
        return f"{int(hours)}h {int(minutes)}m {seconds}s {int(milliseconds)}ms"

    def tokenize_and_summon(
        message_content: str,
        tokenizer=tokenizer,
        model=model,
        max_output_length: int = 128,
    ) -> str | None:
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

        logger.trace(f"Input token count: {inputs['input_ids'].shape[1]}")
        if inputs["input_ids"].shape[1] > SAFE_MAX_LENGTH:
            logger.warning(
                f"Input token count ({inputs['input_ids'].shape[1]}) exceeds safe maximum ({SAFE_MAX_LENGTH}). Skipping."  # noqa: E501
            )
            return None

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

        logger.trace("Output generated.")

        logger.debug(
            f"Memory info : {torch.xpu.memory_allocated()} / {torch.xpu.memory_reserved()}"  # noqa: E501
        )

        logger.trace("Decoding output")

        raw_out = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.trace("Postprocessing output (pruning prompt from result)")
        postprocessed_out = raw_out[len(full_prompt) :]

        return postprocessed_out

    def clean_llm_output(output: str | None):
        # Normalize case and spacing
        if not output:
            return None
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

    def get_categories(message: str, max_output_length: int = 25) -> str | None:
        result = tokenize_and_summon(message, max_output_length=max_output_length)
        if result is None:
            logger.warning("tokenize_and_summon returned None for message.")
            return "none"
        cleaned = clean_llm_output(result)
        if cleaned is None:
            logger.warning("clean_llm_output returned None for message.")
            return "none"
        return cleaned

    logger.info("Categorizing messages")

    starting_time = time()
    last_checked_time = starting_time
    processing_times = []
    total_processed = 0
    llm_processed = 0
    # llm_jettisoned = 0

    logger.info(f"Starting categorization process at {display_time(starting_time)}")

    for message_id, message_content in zip(messages_ids, messages_contents):
        message_start_time = time()
        total_processed += 1

        if message_id in blacklist:
            # logger.debug(f"Message {message_id} is blacklisted. Skipping.")
            continue
        elif message_id in categorized_messages:
            # logger.debug(f"Message {message_id} already categorized. Skipping.")
            continue
        else:
            llm_processed += 1
            logger.debug(f"Categorizing message {message_id}")
            categories = get_categories(message_content)
            if categories is None:
                logger.info(f"Message {message_id} was skipped. Blacklisting.")
                with open(BLACKLIST_PATH, "a") as f:
                    f.write(f"{message_id}\n")
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
            average_processing_time = (
                sum(processing_times) / len(processing_times) if processing_times else 0
            )  # noqa: E501
            average_processing_time = int(average_processing_time * 100) / 100
            logger.info(
                f"Time since start: {display_time(message_end_time - starting_time, duration=True)}"  # noqa: E501
            )
            logger.info(
                f"Examined  : {total_processed} messages out of {len(messages_ids)} ({int(total_processed/len(messages_ids)*10000)/100}%)"  # noqa: E501
            )
            logger.info(
                f"Processed : {llm_processed} messages"  # noqa: E501
            )
            # logger.info(
            #     f"Jettisoned: {llm_jettisoned} messages."  # noqa: E501
            # )
            logger.info(
                f"Average processing time: {average_processing_time} seconds"  # noqa: E501
            )
            logger.info(
                f"ETA: {display_time(average_processing_time * (len(messages_ids) - total_processed), duration=True)}"  # noqa: E501
            )
            last_checked_time = message_end_time

    logger.success("Categorization process completed.")
    logger.info(f"Total messages examined    : {total_processed}")
    logger.info(f"Total messages categorized : {llm_processed}")
    # logger.info(f"Total messages jettisoned  : {llm_jettisoned}")
    logger.info(
        f"Total processing time      : {display_time(time() - starting_time, duration=True)}"  # noqa: E501
    )
    logger.info(
        f"Average processing time    : {sum(processing_times) / len(processing_times) if processing_times else 0:.2f} seconds"  # noqa: E501
    )
    logger.success("Shutting down.")
    return None


# ###################################################################################
# III. UTILS AND PROMPTS

# III.1 Prompts
# logger.info("Setting up prompts")

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

log_line_regex = (
    r"\d{4}-\d{2}-\d{2}T(\d{2}:\d{2}:\d{2}\.\d{6})\+\d{4}\s.\s([A-Z]+)\s*.\s\w*:[^:]*:\d*\s-\s(.*)"  # noqa: E501
)
log_line_hunt_id = r"Categorizing message ([0-9a-f]{64})"


def parse_log_line(line: str) -> tuple[str, str]:
    """Parses a line from the log file"""
    parsed = re.match(log_line_regex, line)
    if parsed is None:
        logger.error(f"Failed to parse log line: {line}")
        raise ValueError(f"Failed to parse log line: {line}")
    return parsed.groups()[0], parsed.groups()[1]


def check_for_timeouts(LOG_FILE: str, timeout: int = 30) -> None | str:
    """Reads the log files to detect subprocess stalling,
    and if so returns the faulty message's id"""
    with open(LOG_FILE, "r") as f:
        log_content = f.readlines()
    last_log = log_content[-1]
    str_last_time = parse_log_line(last_log)[0]
    date_last_time = datetime.datetime.strptime(str_last_time, "%H:%M:%S.%f")
    if (datetime.datetime.now() - date_last_time).seconds > timeout:
        # The last log entry is older than the timeout
        id_match = re.search(log_line_hunt_id, log_content[-4])
        if id_match:
            return id_match.groups()[0]
        else:
            logger.warning("Did not identify the faulty message from logs. Please check formatting")
            logger.warning("Program will assume non-fatal stalling.")
            return "[NotAnId]"
            # raise ValueError(
            #     "Did not identify the faulty message from logs Please check formatting."  # noqa: E501
            # )
    else:
        return None


# ###################################################################################
# IV. CATEGORIZATION

if __name__ == "__main__":
    # II. SETUP
    from utils import log_setup

    LOG_FILE = log_setup.LOG_FILE

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

    logger.info("Setting up persistent categorization")
    non_faulty_stalls = 0
    while True:
        logger.info("Loading messages from database")
        raw_messages = mess_collection.get()
        messages_ids = raw_messages["ids"]
        messages_contents = raw_messages["documents"]
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

        logger.info(f"Starting subprocess - logging to {LOG_FILE}")

        categorization_process = Process(
            target=categorization_loop,
            args=(
                MODEL_PATH,
                MESSAGES_CATEGORIES_PATH,
                LOG_FILE,
                BLACKLIST_PATH,
                prompt_begin,
                prompt_end,
                messages_ids,
                messages_contents,
                categorized_messages,
                blacklist,
            ),
        )
        categorization_process.start()

        logger.info("Subprocess up and running. Watching for stalling.")
        terminated = False
        while categorization_process.is_alive():
            sleep(10)
            faulty_id = check_for_timeouts(str(LOG_FILE))
            if faulty_id:
                if faulty_id == "[NotAnId]":
                    non_faulty_stalls += 1
                    if non_faulty_stalls > 4:
                        logger.error(
                            "Subprocess stalled multiple times without identifying the faulty message. Killing."  # noqa: E501
                        )
                        categorization_process.terminate()
                        terminated = True
                        break
                    else:
                        logger.warning(
                            f"Subprocess stalled {non_faulty_stalls} times without identifying the faulty message. Waiting."  # noqa: E501
                        )
                        continue
                else:
                    logger.error(f"Subprocess stalled on message {faulty_id}. Killing.")
                    logger.info(f"Blacklisting message {faulty_id}")
                    with open(BLACKLIST_PATH, "a") as f:
                        f.write(f"{faulty_id}\n")
                    categorization_process.terminate()
                    terminated = True
                    break
            else:
                non_faulty_stalls = 0
                continue

        if not terminated:
            logger.info("Subprocess terminated by itself. Checking for completion.")
            if not categorization_process.exitcode == 0:
                logger.error("Subprocess terminated with an error")
                break
            else:
                logger.success("Subprocess completed successfully.")
                break

    logger.info("Shutting down.")
