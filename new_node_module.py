from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Updated ChatOpenAI import per deprecation notice
from langchain_openai import ChatOpenAI  # corrected import
from langchain.schema import HumanMessage, AIMessage, SystemMessage, Document
from langchain.prompts import PromptTemplate
from langchain import hub
from pydantic import BaseModel, Field

# LangGraph and related utilities
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# Document retrieval & vectorstore libraries
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import OllamaEmbeddings

# Updated output parser import (new location)
from langchain_core.output_parsers import StrOutputParser
