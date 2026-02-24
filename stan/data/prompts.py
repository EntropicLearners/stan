"""
Prompts used for LLMS
=====================

This module contains a single variable named `prompts` that stores templates 
for prompts to be used with various LLM models. We currently have the following
options available:
- `default` - Prompt suitable for LLaMa models
- `gemma3-270m` - Prompt for the Gemma models from Google
- `short-context` - Prompt suitable for tiny models with small context and fast results.
"""
from string import Template

prompts = {
    "default": (
        "",
        Template('''You are an expert research assistant. You are given page numbers (CONTEXT) in Stanley I. Sandler's book, Chemical, Biochemical, and Engineering Thermodynamics, 5th edition for a TOPIC a student needs help with. Instructions: - The CONTEXT consists of page numbers and associated sub-topics. - Base your response *only* on the TOPIC using the CONTEXT. - Do not invent or generalize; refer to specific passages in the CONTEXT only. - Do not make anything up.   

            Format your answer in two parts:   1. Summarize the the sub-topics in a few sentences.  2. Make a table of the sub-topics.  

            TOPIC:
            $query
            
            CONTEXT in JSON format:
            $retrieved_knowledge

            Now provide a concise, accurate answer to the TOPIC based on the CONTEXT. If no relevant information is found, say "No relevant information found in the index."
        ''')
    ),
    "gemma3-270m": (
        "",
        Template('''Look at this topic and the book pages provided.
        TOPIC: $query
        BOOK PAGES with TOPIC description: $retrieved_knowledge
        Task: Tell the student which pages in Stanley I. Sandler's 5th edition textbook contain information about their topic. Remove unnecessary formatting and make the text look nice for terminal display.
        ''')
    ),
    "short-context": (
        "",
        Template('''You are given page numbers in Stanley I. Sandler's book, Chemical, Biochemical, and Engineering Thermodynamics, 5th edition for a TOPIC a student needs help with. 
            Instructions: - The CONTEXT consists of page numbers and associated sub-topics in JSON format. Base your response *only* using the CONTEXT. - Do not invent or generalize; refer to specific passages in the CONTEXT only. - Do not make anything up.   
            
            TOPIC: $query
            CONTEXT in JSON format: $retrieved_knowledge

            Now provide a concise, accurate answer to the TOPIC based on the CONTEXT. If no relevant information is found, say "No relevant information found in the index."
        ''')
    ),
    "old": (
        "",
        Template('''You are an expert research assistant. You are given page numbers (CONTEXT) in Stanley I. Sandler's book, Chemical, Biochemical, and Engineering Thermodynamics, 5th edition for a TOPIC a student needs help with.

            Instructions:
            - The CONTEXT consists of page numbers and associated sub-topics.
            - Base your response *only* on the TOPIC using the CONTEXT.
            - Do not invent or generalize; refer to specific passages in the CONTEXT only.
            - Do not make anything up.

            Format your answer in two parts:

            1. Summarize the the sub-topics in a few sentences.

            2. Make a list of the pages and sub-topics. The format for each should be:  
            <page range> - <content description>

            TOPIC:
            $query
            
            CONTEXT:
            $retrieved_knowledge

            Now provide a concise, accurate answer to the TOPIC based on the CONTEXT. If no relevant information is found, say "No relevant information found in the index."
        ''')
    )
}