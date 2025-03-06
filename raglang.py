##############################
# 1. Imports & Logging Setup #
##############################

import os
import logging
from typing import Annotated, Sequence, Literal, Union, Dict, Any
from typing_extensions import TypedDict
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
# Added missing imports for conversation history persistence:
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
# New imports for image generation
from g4f.client import Client
import json
from enum import Enum

###############################
# 1. Initialize Embeddings    #
###############################
# Add immediately after imports
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
console = Console()

def log_info(msg: str):
    console.print(f"[bold blue]➡️ {msg}[/]")
    logging.info(msg)

def log_success(msg: str):
    console.print(f"[bold green]✅ {msg}[/]")
    logging.info(f"SUCCESS: {msg}")

def log_error(msg: str):
    console.print(f"[bold red]❌ {msg}[/]")
    logging.error(f"ERROR: {msg}")


#############################################
# 2. Conversation History Persistence Setup  #
#############################################

log_info("Initializing conversation history store")
try:
    conversation_store = Chroma.from_documents([], embedding=embeddings, collection_name="conversation-history")
except ValueError:
    log_info("Empty conversation history; initializing new collection")
    conversation_store = Chroma(embedding_function=embeddings, collection_name="conversation-history")

def get_global_history_count() -> int:
    results = conversation_store.similarity_search("", k=10000)
    return len(results)

def save_conversation_message(message: str, role: str):
    doc = Document(page_content=message, metadata={"role": role})
    conversation_store.add_documents([doc], ids=[str(uuid4())])
    count = get_global_history_count()
    log_info(f"Saved message for '{role}'. Global chat history count: {count}")

def retrieve_conversation_history(query: str, k: int = 3) -> str:
    results_with_scores = conversation_store.similarity_search_with_score(query, k=k)
    log_info(f"Similarity search returned {len(results_with_scores)} results")
    retrieved_messages = []
    for i, (doc, score) in enumerate(results_with_scores, start=1):
        if doc.metadata.get("role", "").lower() == "human":
            preview = doc.page_content[:50].replace("\n", " ")
            log_info(f"Found user msg {i}: '{preview}...' (Score: {score:.4f})")
            retrieved_messages.append(doc.page_content)
        else:
            log_info(f"Skipped non-user msg (role: {doc.metadata.get('role')}) (Score: {score:.4f})")
    if not retrieved_messages:
        log_info("No similar user messages found")
    return "\n".join(retrieved_messages)

def display_chat_history_status(session_state: dict):
    global_count = get_global_history_count()
    session_count = len(session_state.get("messages", []))
    log_info(f"Chat Status: Global count = {global_count}; Session count = {session_count}")


###############################
# 2. Tool Definitions         #
###############################

class ImageModel(str, Enum):
    MIDJOURNEY = "midjourney"
    DALLE3 = "dall-e-3"
    FLUX_PRO = "flux-pro"
    FLUX_DEV = "flux-dev"
    FLUX = "flux"

def generate_image(prompt: str, model: str = "midjourney") -> str:
    """Generate an image based on prompt and return the URL."""
    log_info(f"Generating image with prompt: '{prompt}' using model: {model}")
    try:
        client = Client(base_url="http://localhost:15205/v1")
        response = client.images.generate(
            model=model,
            prompt=prompt,
            response_format="url"
        )
        image_url = response.data[0].url
        log_success(f"Image generated successfully: {image_url[:50]}...")
        return image_url
    except Exception as e:
        log_error(f"Error generating image: {str(e)}")
        return f"Error generating image: {str(e)}"

#####################################################################
# 4. Agent State & Memory Optimization                              #
#####################################################################

class ToolCall(TypedDict):
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: str

class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]
    tool_calls: list[ToolCall]
    requires_tool: bool
    current_tool: Union[str, None]

def agent(state: AgentState):
    log_info("Invoking agent LLM")
    messages = state["messages"]
    sys_inst = """You are a helpful assistant with access to tools.
    
If the user asks to generate an image or create any kind of visual, identify this as an image generation request.
DO NOT attempt to generate images yourself. Instead, indicate that you'll use the ImageGenerator tool.

Available tools:
- ImageGenerator: Generates images based on a text prompt. Provide a detailed prompt and specify a model from: midjourney, dall-e-3, flux-pro, flux-dev, flux.

Example image request detection:
- "Create an image of a cat" → Use ImageGenerator
- "Draw me a landscape" → Use ImageGenerator
- "Visualize a futuristic city" → Use ImageGenerator
"""
    if not messages or getattr(messages[0], "role", "").lower() != "system":
        messages.insert(0, SystemMessage(content=sys_inst))
        log_info("System instruction injected")
    
    model = ChatOpenAI(
        base_url="http://localhost:15205/v1",
        model_name="gemini-2.0-flash",
        temperature=0.5,
        api_key="324"
    )
    
    response = model.invoke(messages)
    log_success("Agent returned direct response")
    return {"messages": [response], "requires_tool": False, "current_tool": None}

def route_tool(state: AgentState) -> Union[Literal["image_tool"], Literal["agent"]]:
    """Determine if we need to use a tool based on the agent's response."""
    messages = state["messages"]
    if not messages:
        return "agent"
    
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return "agent"
    
    content = last_message.content.lower()
    
    image_keywords = [
        "image generator", "generate an image", "create an image", 
        "generate image", "imagegenerat", "drawing", "generate a picture", 
        "create a visual", "visualize"
    ]
    
    if any(keyword in content for keyword in image_keywords):
        log_info("Image generation request detected")
        return "image_tool"
    
    return "agent"

def process_image_tool(state: AgentState) -> AgentState:
    """Process image generation request."""
    messages = state["messages"]
    last_ai_message = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    last_human_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    
    if not last_ai_message or not last_human_message:
        log_error("Could not find necessary messages for image generation")
        return state
    
    # Extract prompt and model from messages
    content = last_ai_message.content + " " + last_human_message.content
    
    # Simple extraction logic for prompt and model
    prompt = content
    model = "midjourney"  # Default model
    
    for m in ImageModel:
        if m.value in content.lower():
            model = m.value
            break
    
    # Generate image
    image_url = generate_image(prompt, model)
    
    # Prepare result message
    result_message = f"I've generated this image for you: {image_url}\n\nUsing model: {model}\nPrompt: {prompt}"
    
    # Create tool call record
    tool_call = {
        "tool_name": "ImageGenerator",
        "tool_input": {"prompt": prompt, "model": model},
        "tool_result": image_url
    }
    
    # Update state with the tool result
    tool_calls = state.get("tool_calls", [])
    tool_calls.append(tool_call)
    
    # Add the result as a new AI message
    new_message = AIMessage(content=result_message)
    
    return {
        "messages": state["messages"] + [new_message],
        "tool_calls": tool_calls,
        "requires_tool": False,
        "current_tool": None
    }

def optimize_memory(state: AgentState) -> AgentState:
    THRESHOLD = 10
    LAST_N = 5
    if len(state["messages"]) > THRESHOLD:
        conversation = "\n".join([msg.content for msg in state["messages"] if getattr(msg, "role", "").lower() != "system"])
        summary_prompt = f"Summarize the following conversation briefly:\n\n{conversation}"
        summarizer = ChatOpenAI(
            base_url="http://localhost:15205/v1",
            model_name="gemini-1.5-flash",
            temperature=0.3,
            api_key="324"
        )
        summary_response = summarizer.invoke([HumanMessage(content=summary_prompt)])
        summary_text = summary_response.content
        log_info(f"Conversation summary: {summary_text}")
        new_messages = [SystemMessage(content="Conversation Summary: " + summary_text)]
        new_messages.extend(state["messages"][-LAST_N:])
        log_info(f"Session history optimized. New session count: {len(new_messages)}")
        return {
            "messages": new_messages,
            "tool_calls": state.get("tool_calls", []),
            "requires_tool": state.get("requires_tool", False),
            "current_tool": state.get("current_tool", None)
        }
    
    log_info(f"Session history count remains: {len(state['messages'])}")
    return state


####################################################################
# 5. Graph Building & MongoDB Persistence (Permanent Storage)      #
####################################################################

log_info("Building state graph with image generation capability")
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("image_tool", process_image_tool)

# Add conditional routing
workflow.add_conditional_edges(
    "agent",
    route_tool,
    {
        "image_tool": "image_tool",
        "agent": END
    }
)

workflow.add_edge(START, "agent")
workflow.add_edge("image_tool", END)

from langgraph.checkpoint.mongodb import MongoDBSaver
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
log_info(f"Connecting to MongoDB at {MONGODB_URI}")

with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)
    log_success("State graph compiled with MongoDB persistence")

    ####################################################################
    # 6. Interactive Chat Loop with Full Persistence                   #
    ####################################################################

    config = {"configurable": {"thread_id": "main_thread"}}
    state: AgentState = {"messages": [], "tool_calls": [], "requires_tool": False, "current_tool": None}

    console.clear()
    console.print("[bold blue]🤖 Chat Interface with Image Generation[/]\n", justify="center")
    console.print("Type 'exit' to end the conversation\n")

    while True:
        user_input = console.input("[bold green]User: [/")
        if user_input.lower() == "exit":
            break

        state["messages"].append(HumanMessage(content=user_input))
        save_conversation_message(user_input, role="human")
        state = optimize_memory(state)
        inputs = {
            "messages": state["messages"], 
            "tool_calls": state.get("tool_calls", []),
            "requires_tool": False,
            "current_tool": None
        }

        response_text = ""
        console.print("\n[bold blue]Agent:[/]")
        console.print("[dim]Starting graph execution...[/dim]")
        
        with console.status("[bold yellow]Processing...[/]"):
            outputs = {}
            for output in graph.stream(inputs, config=config):
                outputs.update(output)
                if "agent" in output:
                    msgs = output.get("agent", {}).get("messages", [])
                    if msgs and isinstance(msgs[0], AIMessage):
                        response_text = msgs[0].content
                if "image_tool" in output:
                    msgs = output.get("image_tool", {}).get("messages", [])
                    if msgs and isinstance(msgs[-1], AIMessage):
                        response_text = msgs[-1].content

        if response_text:
            # Check if response contains an image URL
            if "http" in response_text and any(ext in response_text.lower() for ext in [".jpg", ".png", ".jpeg", ".gif"]):
                # Extract the URL for special formatting
                console.print(Panel(response_text, border_style="green", title="Image Generated"))
            else:
                console.print(Panel(response_text, border_style="blue", title="Response"))
            
            # Update state with the final response
            if not any(msg.content == response_text for msg in state["messages"] if isinstance(msg, AIMessage)):
                state["messages"].append(AIMessage(content=response_text))
            save_conversation_message(response_text, role="agent")
        else:
            console.print(Panel("No response generated.", border_style="red", title="Error"))
        
        # Update tool calls from output
        if "tool_calls" in outputs.get("image_tool", {}):
            state["tool_calls"] = outputs["image_tool"]["tool_calls"]
        
        display_chat_history_status(state)
        console.print()

    log_info("Exiting chat session. Closing MongoDB connection.")
