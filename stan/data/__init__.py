"""
Stan Data Directory
===================

Houses data that is used in the query and rag models

* `bindex_tab_flat.txt` - Flattened text file with index of the book
* `bindex_tab.json` - Hierarchical JSON file with the index of the book
* `bindex_tab.txt` - Hierarchical text file with the index of the book
* `ftoc_nav_tree.json` - Hierarchical JSON file with the table of contents of the book

Additionally the module also includes some functions to work with LLMs and the data.

JSON index structure used by `bindex_tab.json`
----------------------------------------------
The JSON structure root is an array. Each element of the array is is a topic object.
The subtopics field can contain more topics. The topics-subtopics structure is recursive.

JSON schema:

.. code-block:: json

   {
     "$schema": "https://json-schema.org/draft/2020-12/schema",
     "title": "Book Index",
     "description": "Schema for representing a hierarchical book index with topics, subtopics, and page ranges.",
     "type": "array",
     "items": {
       "$ref": "#/$defs/topic"
     },
     "$defs": {
       "topic": {
         "type": "object",
         "properties": {
           "topic": {
             "type": "string",
             "description": "The title of the topic."
           },
           "pages": {
             "type": "string",
             "description": "The page number or range where the topic is discussed (e.g., '799–817').",
             "pattern": "^[0-9]+(–[0-9]+)?(,[0-9]+(–[0-9]+)?)*$"
           },
           "subtopics": {
             "type": "array",
             "description": "List of subtopics related to this topic.",
             "items": {
               "$ref": "#/$defs/topic"
             }
           }
         },
         "required": ["topic"],
         "additionalProperties": false
       }
     }
   }

"""

from .prompts import prompts