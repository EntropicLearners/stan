# thermo_rag_tab.py
# Run a querry on a simple JSON index
# This doesn't actually need llamaindex
# 
# E. M. Furst, September 2025


import os
import json
import random
import ollama
import pathlib

from ..data import prompts
from ..utils import fun


#
# Functions and routines
#
def load_index(json_path):
    """
    Index parser
    Reads structured index and creates python dictionary
    """
    stan_root_dir = pathlib.Path(__file__).resolve().parent.parent
    json_path = stan_root_dir / json_path
    print(stan_root_dir)
    print(json_path)
    if os.path.exists(json_path):
        print(f"Loading cached index from {json_path}...")
        with open(json_path, "r") as f:
            return json.load(f)
    else:
        print("Cached index not found. Please create JSON index first.")

        return None


def lookup_topic(index, query):
    """
    Simple lookup function -- exact or partial match
    """
    query = query.lower().strip()
    results = []

    # Exact match
    if query in index:
        results.extend(index[query])

    # Partial matches
    for key in index.keys():
        if query in key and key != query:
            results.extend(index[key])

    return results

def search_book_index(index_data, query: str) -> list:
    """
    Search index based on query passed from user
    """
    query = query.lower()
    results = []

    for topic in index_data:
        if query in topic["topic"].lower():
            print(topic)
            results.append(topic)

    if not results:
        return []
    
    # output = []
    # for item in results:
    #     context = f" ({item['context']})" if item['context'] else ""
    #     pages = ", ".join(item['pages'])
    #     output.append(f"- Pages: {pages} Sub-topic: {context}")
    # return "\n".join(output)
    print("Results here: ", results)
    return results


# 
# Main program routine
#
def main(index_data, logger, llm, prompt_option):

    # Run the query
    while True:
        query = input("\nEnter a topic to look up (or 'exit'): ").lower()
        if query == 'exit' or query == '':
            break
        print("Let me look that up...")

        # rewrite query?

        # Simple search / match
        retrieved_knowledge = search_book_index(index_data, query)

        if retrieved_knowledge == []:
            print (f"No direct matches found for '{query}'. Try another term.")
            continue
        else:
            logger.info(f"DEBUG For the query '{query}':")
            logger.info(f"DEBUG Retrieved knowledge:\n{retrieved_knowledge}\n\n")

            # Compose response
            system_prompt, instruction_prompt = prompts[prompt_option]
            instruction_prompt = instruction_prompt.substitute(query=query, retrieved_knowledge=retrieved_knowledge)

            stream = ollama.chat(
                model=llm, 
                messages=[
                    {'role': 'system', 'content': f"{system_prompt}"}, 
                    {'role': 'user', 'content': f"{instruction_prompt}"},
                ],
                stream=True,
            )
            # print the response from the chatbot in real-time
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            print("\n")

            # Bonus poem
            r = random.random()
            if r > 0.66:
                fun.generate_poem(system_prompt, query, retrieved_knowledge,llm)
            elif r > 0.33 :
                fun.generate_haiku(system_prompt, query, retrieved_knowledge, llm)
            else:
                pass

