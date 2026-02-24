import ollama

def generate_poem(system_prompt, query, retrieved_knowledge, llm="llama3.1:8B"):
    stream = ollama.chat(
        model=llm, 
        messages=[
            {'role': 'system', 'content': f"{system_prompt}"}, 
            {'role': 'user', 'content': f"Write a five line poem about {query} with words found in: {retrieved_knowledge}"},
        ],
        stream=True,
    )
    # print the response from the chatbot in real-time
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")


def generate_haiku(system_prompt, query, retrieved_knowledge, llm="llama3.1:8B"):
    stream = ollama.chat(
        model=llm, 
        messages=[
            {'role': 'system', 'content': f"{system_prompt}"}, 
            {'role': 'user', 'content': f"Write a single haiku about {query} with words found in {retrieved_knowledge}."},
        ],
        stream=True,
    )
    # print the response from the chatbot in real-time
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")