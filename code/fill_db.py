import json

import chromadb
from loguru import logger
from utils import conversation_parsing_tools as cpt
from utils import hash_utils as hash
from utils import load_paths as lp
from utils import log_setup  # noqa: F401
from utils import metadata_prepare as metaprep

# Initialize the paths
logger.debug("Initializing paths")
PATHS = lp.get_paths()
CHROMADB_PATH = PATHS["chroma_db_path"]
SOURCE_CONVERSATIONS_PATH = PATHS["source_conversations_path"]

# Opening persistent database
logger.info("Accessing persistent database")
client = chromadb.PersistentClient(str(CHROMADB_PATH))
logger.success("Persistent database accessed")
logger.debug("Fetching collections")
conv_collection = client.get_collection("conversations")
mess_collection = client.get_collection("messages")
logger.success("Collections fetched")


# Opening JSON source file
logger.debug("Opening JSON source file")
data = json.load(open(SOURCE_CONVERSATIONS_PATH, "r"))

# Parse the JSON file
logger.info("Updating database with currently available data export")
logger.debug("Parsing JSON file")
for conv in data:
    title, messages = cpt.parse_conversation(conv)
    conv_id = hash.hash_set_length(title.encode("utf-8"), length=16)
    logger.trace(f"Conversation ID: {conv_id}")

    logger.debug(f"Inserting conversation: {title}")
    conv_collection.upsert(conv_id, documents=[title])

    logger.debug(f"Inserting messages for conversation: {title}")
    messages_ids = [
        hash.hash_message(message[0] + str(message[1])) for message in messages
    ]
    messages_documents = [message[0] for message in messages]
    messages_metadata = [
        metaprep.cast_metadata(message[1], conv_id) for message in messages
    ]

    mess_collection.upsert(
        messages_ids, documents=messages_documents, metadatas=messages_metadata
    )
    logger.success(f"Conversation {title} inserted")
logger.success("All conversations inserted")
logger.success("Process complete")
