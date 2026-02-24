import sys
import os
import logging
import ollama

sys.path.append('../../')
sys.path.append('../../stan')

import stan


"""
Globals
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LLM model to use in query transform and generation
# llm="command-r7b"
llm="llama3.1:8B"
# llm = "gemma3:1b"
# llm = "gemma3:270m"

# JSON data path (will need to generalize later for multiple files)
json_path = "data/bindex_tab.json"


# Configure the basic logging setup
# This will log messages to the console and a file named 'app.log'
# Messages with a level of INFO or higher will be recorded
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('app.log') # Output to file
    ]
)

# Get a logger instance for the current module
logger = logging.getLogger(__name__)


# index data ends up as a global
# would be better to make a class
index_data = stan.hybrid_query.json_rag_tab.load_index(json_path) 
if index_data is None:
    print("Exiting.")
    exit(1)
else :
    print(f"Index loaded with {len(index_data)} main entries.")    

stan.hybrid_query.json_rag_tab.main(index_data, logger, llm, prompt_option='default')
