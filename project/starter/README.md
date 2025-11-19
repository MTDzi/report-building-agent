# Document Assistant Project Instructions

Welcome to the Document Assistant project!

This project was about implementing a document processing system using LangGraph, in which an AI assistant has access to a set of documents and can: answers questions, summarize documents, and perform calculations. 

## Project Overview

This document assistant uses a multi-agent architecture with LangGraph to handle different types of user requests:
- **Q&A Agent**: Answers specific questions about document content
- **Summarization Agent**: Creates summaries and extracts key points from documents
- **Calculation Agent**: Performs mathematical operations on document data

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
cd <repository_path>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Assistant

```bash
python main.py
```

## Project Structure
```
doc_assistant_project/
├── src/
│   ├── schemas.py        # Pydantic models
│   ├── retrieval.py      # Document retrieval
│   ├── tools.py          # Agent tools
│   ├── prompts.py        # Prompt templates
│   ├── agent.py          # LangGraph workflow
│   └── assistant.py      # Main agent
├── sessions/             # Saved conversation sessions
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md             # This file
```



## Agent Architecture

The LangGraph agent follows this workflow:

![](./docs/langgraph_agent_architecture.png)


## Implementation details

### `main.py`
The `main.py` script's responsibilities are:
1. Initialize the `assistant: DocumentAssistant` with the API key found in the `.env` file.
2. Ask for the user name to create a session.
3. Ask for user message and pass it to the `assistant`.
4. Get the resulting response of the `assistant` and print it out to the standard output.


### `assistant.py`
In the `assistant.py` module is fully dedicated to the implementation of the `DocumentAssistant` class.

It's main method is the `process_message` which:
1. Initializes the config (with the `thread_id` for keeping track of the interaction, `llm`, and `tools`)
2. Create the `initial_state: AgentState` and pass it to the LangGraph workflow.
   * This `initial_state` contains, among others, the conversation history, the documents accessed thus far, and conversation history -- all of which are kept track of by the `DocumentAssistant` object.
3. Get the final state of the `AgentState` after it passes through the whole workflow.
   * If it's a legit final state (it does contain a list of messages exchanged between the user and the AI), we update the conversation history, the current session, and active documents, and we store the session.

The saving of the session is worth taking a moment to appreciate. It is saved as a JSON file, and since not all elements of the session are JSON-compliant (like the `datetime`), we pass a `default` argument to `json.dump`:
```python3
json.dump(session_dict, f, indent=2, default=serialize_datetime)
```
where `seralize_datetime` is responsible for turning a Python `datetime` into a string of an ISO format that can be storred in a JSON.

Now, I was wondering: "How is this being de-serialized? Isn't it a bit problematic to have this decoding done both ways: when we save to JSON and when we load from a JSON? If we'd want to modify something about how we serialize the data, we need to also remember to modify the part where we deserialize the data."

But the deserialization is done in a very clever way: 
```python3
SessionState(**data_from_a_json)
```
and `SessionState` inherits from a Pydantic `BaseModel` and looks like this:
```python3
class SessionState(BaseModel):
    """Session state"""
    session_id: str
    user_id: str
    conversation_history: List[TypedDict] = Field(default_factory=lambda: list)
    document_context: List[str] = Field(default_factory=lambda: list, description="Active document IDs")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
```
which means we delegate the responsibility of deserializing the JSON attributes to appropriate types to Pydantic. I really liked this part!

### `agent.py`
This module contains the definition of the LangGraph workflow, with its main function `create_workflow`.

I ran into this problem that the `AgentState` (which flows through the workflow) was updated with an object of an unapproapriate type and it was hard to find where that actually happens. But `AgentState` inherits from `TypeDict` instead of Pydantic's `BaseModel`. I realized that modifying it to be a subclass of `BaseModel` might be a lot of work, so I settled upon an intermediate solution: I created a sibling class `AgentStateValidator` which is a subclass of `BaseModel`, but I use it only to call its class method:
```python3
AgentStateValidator.model_validate(state)
```
This way I can be sure that the most relevant fields in the `AgentState` are of appropriate types (specified in `AgentStateValidator`).

But how to make sure this `model_validate` class method is called in the functions comprising the graph's nodes? For that I wrote a decorator `agent_state_validator` that I used to, well, decorate all functions used in the graph.

# Summary
The project is a great beginner example of what LangGraph can achieve with a fairly minimal workflow:

1. It can receive a message from a user that is then interpreted (by classifying the intent) as a QnA, a summarization, or a calculation task.

2. It uses four distinct tools: the calculator, the document search, the document reader, and the document statistics tools.

3. It uses schemas to ask the LLM for a structured output for ease of parsing and interpreting the output.

4. It uses state validation to ensure the nodes of the graph don't update the state with values that don't match the expected types.