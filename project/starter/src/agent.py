from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal, Callable, TypeAlias
from functools import wraps
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, conlist, ValidationError
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
from schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT


class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str  # Next node to execute in the graph

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]  # List of document IDs being currently discussed

    # Current task state
    current_response: Optional[Dict[str, Any]]  # The response being built
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]

    # List of agent nodes executed
    actions_taken: Annotated[List[str], operator.add]


class AgentStateValidator(BaseModel):
    """Temporary model to validate initial state runtime types."""
    
    # Validate the types that caused the bug:
    intent: Optional[UserIntent] 
    
    # Validate other standard types if desired (optional)
    user_input: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
    
    # You can skip the LangGraph-specific fields (messages, actions_taken) 
    # as their validation is handled by LangGraph's internal checks later.
    # However, if you include them, they must match the type hints:
    messages: conlist(Any, min_length=0) # Or just list[Any]
    tools_used: List[str]


NodeFunction: TypeAlias = Callable[[AgentState, RunnableConfig], AgentState]


def agent_state_validator(node_function: NodeFunction) -> NodeFunction:
    """
    Decorator that validates the LangGraph state (the first argument) 
    using the AgentStateValidator Pydantic model before executing the node function.
    """

    @wraps(node_function)
    def wrapper(state: AgentState, config: RunnableConfig) -> AgentState:
        if config.get("configurable", {}).get("force_validation") is True:
            try:
                # Use the validator model you created earlier
                AgentStateValidator.model_validate(state)                 
            except ValidationError as e:
                # Halt execution or log the error and transition to a fallback node
                print(f"Validation Error: State failed Pydantic checks in node {node_function.__name__}: {e}")
                # Raising an exception will typically halt the LangGraph execution for debugging
                raise

        return node_function(state, config)

    return wrapper


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> (
Dict[str, Any], List[str]):
    llm_with_tools = llm.bind_tools(
        tools
    )

    agent = create_react_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    tools_used = [t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)]

    return result, tools_used


@agent_state_validator
def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = state.get("messages", [])

    # Configure the llm chat model for structured output
    llm_with_structured_output = llm.with_structured_output(UserIntent)

    # Create a formatted prompt with conversation history and user input
    prompt_template = get_intent_classification_prompt()

    # Call the LLM
    response = llm_with_structured_output.invoke(prompt_template.invoke({
        'user_input': state['user_input'],
        # TODO: Check if works if there is a default_factory
        'conversation_history': state['messages'],
        # 'conversation_history': state.get('messages', []),
    }))

    next_step = {
        'qa': 'qa_agent',
        'summarization': 'summarization_agent',
        'calculation': 'calculation_agent',
    }.get(response.intent_type, 'qa_agent')

    return {
        "actions_taken": ["classify_intent"],
        "intent": response,
        "next_step": next_step,
    }


@agent_state_validator
def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("qa")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        # TODO: Check if works if there is a default_factory
        "messages": result["messages"],
        # "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


@agent_state_validator
def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("summarization")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state["messages"], # TODO state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        # TODO: Check if works if there is a default_factory
        "messages": result["messages"],
        # "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


@agent_state_validator
def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("calculation")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state["messages"], # TODO state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        # TODO: Check if works if there is a default_factory
        "messages": result["messages"],
        # "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


@agent_state_validator
def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    llm = config.get("configurable").get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        # TODO: Check if works if there is a default_factory
        "chat_history": state["messages"],
        # "chat_history": state.get("messages", []),
    })

    structured_llm = llm.with_structured_output(SummarizationResponse)

    response = structured_llm.invoke(prompt_with_history)

    return {
        "conversation_summary": response.summary, # response["summary"],
        "active_documents": response.document_ids,
        "next_step": END,
    }


def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")


def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)
    workflow.add_node("classify_intent", classify_intent)
    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent", 
            "end": END,
        }
    )

    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")
    
    workflow.add_edge("update_memory", END)

    return workflow.compile(checkpointer=InMemorySaver())
