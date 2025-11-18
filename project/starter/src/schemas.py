from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, TypedDict
from datetime import datetime
from enum import Enum


class DocumentChunk(BaseModel):
    """Represents a chunk of document content"""
    doc_id: str = Field(description="Document identifier")
    content: str = Field(description="The actual text content")
    metadata: Dict[str, Any] = Field(default_factory=lambda: dict, description="Additional metadata")
    relevance_score: float = Field(default=0.0, description="Relevance score for retrieval")


class AnswerResponse(BaseModel):
    """
    Structured response for Q&A tasks.
    This schema ensures consistent formatting of answers and tracks which documents were referenced.
    """
    question: str = Field(description="The question asked by the user")
    answer: str = Field(description="The generated answer")
    sources: List[str] = Field(default_factory=lambda: list, description="List of source document IDs user")
    confidence: float = Field(description="Confidence score between 0 and 1") #, ge=0.0, le=1.0)
    timestamp: datetime = Field(description="When the response was generated", default_factory=datetime.now)


class SummarizationResponse(BaseModel):
    """Structured response for summarization tasks"""
    original_length: int = Field(description="Length of original text")
    summary: str = Field(description="The generated summary")
    key_points: List[str] = Field(description="List of key points extracted")
    document_ids: List[str] = Field(default_factory=lambda: list, description="Documents summarized")
    timestamp: datetime = Field(default_factory=datetime.now)


class CalculationResponse(BaseModel):
    """Structured response for calculation tasks"""
    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The calculated result")
    explanation: str = Field(description="Step-by-step explanation")
    units: Optional[str] = Field(default=None, description="Units if applicable")
    timestamp: datetime = Field(default_factory=datetime.now)


class UpdateMemoryResponse(BaseModel):
    """Response after updating memory"""
    summary: str = Field(description="Summary of the conversation up to this point")
    document_ids: List[str] = Field(default_factory=lambda: list, description="List of documents ids that are relevant to the users last message")


class IntentType(str, Enum):
    QA = "qa"
    SUMMARIZATION = "summarization"
    CALCULATION = "calculation"
    UNKNOWN = "unknown"
    

class UserIntent(BaseModel):
    """
    User intent classification
    This schema helps the system understand what type of request the user is making and route it to the appropriate agent.
    """
    intent_type: Literal["qa", "summarization", "calculation", "unknown"] = Field(
        description="The classified intent ('qa', 'summarization', 'calculation', or 'unknown')"
    )
    # intent_type: IntentType = Field(description="The classified intent ('qa', 'summarization', 'calculation', or 'unknown')")
    confidence: str = Field(description="Confidence in classification (float between 0 and 1)") #, ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation for the classification")


class SessionState(BaseModel):
    """Session state"""
    session_id: str
    user_id: str
    conversation_history: List[TypedDict] = Field(default_factory=lambda: list)
    document_context: List[str] = Field(default_factory=lambda: list, description="Active document IDs")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
