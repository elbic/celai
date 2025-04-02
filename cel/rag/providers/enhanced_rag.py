from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from cel.model.common import ContextMessage
from cel.rag.providers.rag_retriever import RAGRetriever
from cel.rag.stores.vector_store import VectorRegister
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

DEFAULT_PROMPT = """You are an AI assistant specialized in transforming conversation inputs into well-structured and context-aware queries for retrieval-augmented generation (RAG).

Your goal is to analyze the conversation history and refine the user's query to maximize its relevance and effectiveness for vectorization and retrieval, without changing the original intent.

For each query, you should:
1. Extract key entities and concepts
2. Identify the user's intent and context
3. Generate relevant synonyms and related terms
4. Consider temporal context if present
5. Maintain the original query's core meaning

Output should be a JSON object with the following structure:
{
    "enhanced_query": "The refined query",
    "entities": ["List of key entities"],
    "intent": "The user's intent",
    "context": "Relevant context from history",
    "related_terms": ["List of related terms"]
}"""

@dataclass
class QueryRefiner:
    model_name: str = "gpt-3.5-turbo"
    n_history_messages: int = 5  # Increased from 3 to 5 for better context
    custom_prompt: str = DEFAULT_PROMPT
    temperature: float = 0.1  # Slightly increased for more creative query expansion
    max_tokens: int = 150  # Increased to handle more complex queries
    llm: Optional[ChatOpenAI] = field(init=False, default=None)
    output_parser: JsonOutputParser = field(init=False, default=None)

    def __post_init__(self):
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        self.output_parser = JsonOutputParser()

    def _format_history(self, history: List[ContextMessage]) -> str:
        """Format conversation history in a more structured way."""
        if not history:
            return ""
        
        formatted_history = []
        for msg in history[-self.n_history_messages:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted_history.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted_history)

    def enhance_query(self, query: str, history: List[ContextMessage]) -> Dict[str, Any]:
        """Enhanced query refinement with structured output."""
        if not history:
            return {
                "enhanced_query": query,
                "entities": [],
                "intent": "unknown",
                "context": "",
                "related_terms": []
            }

        history_context = self._format_history(history)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.custom_prompt),
            ("human", "History:\n{history}\n\nUser Query: {query}")
        ])

        chain = prompt | self.llm | self.output_parser
        
        try:
            result = chain.invoke({
                "history": history_context,
                "query": query
            })
            return result
        except Exception as e:
            # Fallback to basic query if enhancement fails
            return {
                "enhanced_query": query,
                "entities": [],
                "intent": "unknown",
                "context": "",
                "related_terms": []
            }

class EnhancedRetriever(RAGRetriever):
    def __init__(self, base_retriever: RAGRetriever, query_builder: QueryRefiner = None):
        self.base_retriever = base_retriever
        self.query_builder = query_builder or QueryRefiner()

    def search(self,
               query: str,
               top_k: int = 1,
               history: List[ContextMessage] = None,
               state: dict = {}) -> List[VectorRegister]:
        # Use the QueryBuilder to enhance the query
        enhanced_result = self.query_builder.enhance_query(query, history)
        
        # Use the enhanced query for retrieval
        results = self.base_retriever.search(enhanced_result["enhanced_query"], top_k)
        
        # Add metadata from query enhancement
        for result in results:
            result.metadata.update({
                "enhanced_entities": enhanced_result["entities"],
                "enhanced_intent": enhanced_result["intent"],
                "enhanced_context": enhanced_result["context"],
                "related_terms": enhanced_result["related_terms"]
            })
        
        return results
