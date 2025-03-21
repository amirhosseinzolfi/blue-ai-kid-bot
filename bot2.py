# ============================================================
#                   START OF MODULE IMPORTS
# ============================================================
"""
Refined Telegram Bot using LangChain and LangGraph
---------------------------------------------------
This version addresses issues with short-term, long-term, and persistent memory,
and refines the LangGraph state workflow for improved efficiency and clarity.
"""

import os
import getpass
import logging
from datetime import datetime
from time import time
import re
from typing import Annotated, Sequence, Literal, Union, Dict, Any
from typing_extensions import TypedDict
from telebot.apihelper import ApiTelegramException


# ============================================================
#           RICH CONSOLE & LOGGING SETUP
# ============================================================
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.logging import RichHandler
from rich.pretty import pprint

console = Console()
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for extra verbosity
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["OPENAI_API_KEY"] = "234"
os.environ["TELEGRAM_BOT_TOKEN"] = "8028667097:AAEOQqzrC9r14j1BLF2fWTuh1ZcKpItzFEA"


# ============================================================
#           ENVIRONMENT VARIABLE SETUP
# ============================================================
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
        logger.info(f"Environment variable '{var}' set by user input.")
    else:
        logger.info(f"Environment variable '{var}' already set.")

_set_env("OPENAI_API_KEY")
# Removed the MONGODB_URI prompt
# ...existing code...
# Initialize bot logger globally
from bot_logger import BotLogger
bot_logger = BotLogger()


# ============================================================
#                MONGODB SETUP
# ============================================================
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

def get_mongodb_connection():
    """Establishes connection to MongoDB and returns the client"""
    try:
        # Use hardcoded MongoDB URL instead of environment variable
        mongodb_uri = "mongodb://localhost:27017"  # Define it here
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Verify the connection works
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client, mongodb_uri  # Return BOTH client and URI
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None, None

# Initialize MongoDB client AND URI
mongodb_client, mongodb_uri = get_mongodb_connection()  # Store both return values
if (mongodb_client):
    db = mongodb_client.bluecare_db
    chat_history_collection = db.chat_history
    long_term_memory_collection = db.long_term_memory
    user_settings_collection = db.user_settings
    logger.info("MongoDB collections initialized")
    bot_logger.log_info("Database initialization complete.")  # New log entry
else:
    logger.warning("MongoDB not available, will operate with in-memory storage only")
    bot_logger.log_info("Running with in-memory storage only.")

# ============================================================
#          TELEGRAM BOT IMPORTS & SETUP
# ============================================================
# Removed Telegram bot imports and initialization

# ============================================================
#         LANGCHAIN & LANGGRAPH IMPORTS
# ============================================================
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.mongodb import MongoDBSaver
from enum import Enum
from g4f.client import Client
# Add imports for image-based AI
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import base64
from prompts import get_system_instruction, get_summary_prompt, get_image_prompt, get_info_prompt, get_ai_tone_prompt, DEFAULT_AI_TONE, KIDS_INFO, LOADING_MESSAGE_TEXT, MARKDOWN_V2_PARSE_MODE



# ============================================================
#            MEMORY & EMBEDDINGS SETUP
# ============================================================
#from langmem import create_manage_memory_tool, create_search_memory_tool  # No longer needed

# ------------------------------ LLM and Embeddings Initialization ------------------------------
llm = ChatOpenAI(
    base_url="http://localhost:15205/v1",
    model_name="gpt-4o",
    temperature=0.5,
    api_key="324"
)
logger.info("ChatOpenAI LLM initialized with streaming enabled.")

# New LLM for handling kid info and AI tone
info_llm = ChatOpenAI(
    base_url="http://localhost:15205/v1",
    model_name="gemini-2.0-flash",
    temperature=0.3,
    api_key="324"
)
logger.info("Info LLM initialized.")

# LLM for summarization
summarization_llm = ChatOpenAI(  # Dedicated LLM for summarization
    base_url="http://localhost:15205/v1",
    model_name="gpt-4o",  # Use a faster model
    temperature=0.3,  # Lower temperature for more concise summaries
    api_key="324"
)
logger.info("Summarization LLM initialized.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize a separate model for image-based queries
mm_model = ChatOpenAI(
    base_url="http://localhost:15205/v1",  # Use the same local endpoint
    model_name="gpt-4o",                  # Specify model name consistently
    temperature=0.5,                      # Set appropriate temperature
    api_key="324"                         # Use same API key as other models
)
logger.info("Multimodal LLM initialized for image processing.")

# ============================================================
#         PROMPT TEMPLATES & CONSTANTS
# ============================================================
# Global settings for user-specific configuration
setting_data = {}
STICKER_SETTING = None  # Sticker ID if neededيال


# ============================================================
#              BOT LOGGER CLASS DEFINITION
# ============================================================
# ------------------------------ Utility Functions ------------------------------
def refine_ai_response(text: str) -> str:
    """Cleans up and reformats the AI's output for Telegram MarkdownV2 compatibility,
    preserving **bold** formatting."""
    text = re.sub(r"\[Sticker:.*?\]", "", text)
    # Removed conversion of ** to * so that bold markers remain intact.
    # Instead, convert double underscores to single underscores for italic if needed.
    text = re.sub(r"__([^_]+)__", r"_\1_", text)
    text = re.sub(r"^####\s+(.*?)$", r"▫️\1", text, flags=re.MULTILINE)
    text = re.sub(r"^###\s+(.*?)$", r"🟢 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^##\s+(.*?)$", r"🔶 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^#\s+(.*?)$", r"⭐ \1", text, flags=re.MULTILINE)
    text = re.sub(r"^(?:\s*[-*]\s+)(.*?)$", r"🔹 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^(?:\s*\d+\.\s+)(.*?)$", r"🔹 \1", text, flags=re.MULTILINE)
    # Collapse multiple asterisks to exactly two for bold formatting.
    text = re.sub(r"\*{2,}", r"**", text)
    text = re.sub(r"\.{3,}", "...", text)
    return text.strip()

def escape_markdown_v2(text: str) -> str:
    """Escapes markdown sensitive characters for Telegram using negative lookbehind to avoid double escaping."""
    import re
    return re.sub(r"(?<!\\)([*_\[\]()~`>#+\-=|{}.!])", r"\\\1", text)

# Updated safe_escape_markdown_v2: process markdown links separately.
def safe_escape_markdown_v2(text: str) -> str:
    """
    Escapes MarkdownV2 reserved characters in text, preserving proper markdown link formatting.
    Only the link text is escaped while the URL remains unchanged.
    """
    import re
    # Callback to escape link text only.
    def escape_link(match):
        link_text = match.group(1)
        url = match.group(2)
        return f"[{escape_markdown_v2(link_text)}]({url})"
        
    # Define pattern for markdown links.
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    parts = []
    last_end = 0
    for m in link_pattern.finditer(text):
        # Process text before the markdown link.
        pre_text = text[last_end:m.start()]
        parts.append(escape_markdown_v2(pre_text))
        # Process the markdown link using the callback.
        parts.append(escape_link(m))
        last_end = m.end()
    # Process any trailing text.
    parts.append(escape_markdown_v2(text[last_end:]))
    return "".join(parts)

# New conversion: Replace "**bold**" with "<b>bold</b>"
def convert_bold_markdown_to_html(text: str) -> str:
    import re
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

# Update the function to properly handle bold text for Telegram MarkdownV2
def convert_bold_to_telegram_format(text: str) -> str:
    """
    Converts all forms of bold markdown syntax to properly escaped bold text for Telegram MarkdownV2.
    This properly handles asterisks and preserves links.
    """
    import re
    
    # First, extract and temporarily store all markdown links to protect them
    links = []
    def extract_links(match):
        links.append((match.group(1), match.group(2)))
        return f"LINK_PLACEHOLDER_{len(links)-1}_"
    
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    text_with_placeholders = link_pattern.sub(extract_links, text)
    
    # Convert **bold** to properly escaped bold format: *bold*
    # Note: For Telegram MarkdownV2, bold is single asterisks but needs escaping
    text_with_placeholders = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text_with_placeholders)
    
    # Remove any standalone single asterisks (improper formatting)
    text_with_placeholders = re.sub(r"(?<!\*)\*(?!\*)\s*", "", text_with_placeholders)
    
    # Restore all links
    for i, (link_text, url) in enumerate(links):
        # Handle bold within link text
        link_text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", link_text)
        text_with_placeholders = text_with_placeholders.replace(
            f"LINK_PLACEHOLDER_{i}_", 
            f"[{link_text}]({url})"
        )
        
    return text_with_placeholders

# Initialize bot logger globally
from bot_logger import BotLogger
bot_logger = BotLogger()



# ============================================================
#     LANGGRAPH AGENT RESPONSE & STATE FUNCTIONS
# ============================================================
class ToolCall(TypedDict):
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: str

class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]
    tool_calls: list[ToolCall]
    requires_tool: bool
    current_tool: Union[str, None]
    chat_id: str  # Add chat_id to the state


class ImageModel(str, Enum):
    MIDJOURNEY = "midjourney"
    DALLE3 = "dall-e-3"
    FLUX_PRO = "flux-pro"
    FLUX_DEV = "flux-dev"
    FLUX = "flux"

def generate_image(prompt: str, model: str = "midjourney") -> str:
    """Generate an image based on prompt and return the URL."""
    logger.info(f"Generating image with prompt: '{prompt}' using model: {model}")
    try:
        client = Client(base_url="http://localhost:15205/v1")
        response = client.images.generate(
            model=model,
            prompt=prompt,
            response_format="url"
        )
        image_url = response.data[0].url
        logger.info(f"Image generated successfully: {image_url[:50]}...")
        return image_url
    except Exception as e:
        error_message = f"Error generating image: {str(e)}"
        logger.error(error_message)
        return error_message

def agent(state: AgentState):
    logger.info("Invoking agent LLM")
    messages = state["messages"]
    chat_id = state["chat_id"]  # Get chat_id from the state
    user_memory = get_user_memory(chat_id) #Use chat ID as user ID
    kid_info = user_memory.get_kid_info() or KIDS_INFO
    ai_tone = user_memory.get_ai_tone() or DEFAULT_AI_TONE
    user_name = user_memory.get_user_name() or ""  # Get user name, default to empty string if not set

     # Get the last 5 messages from the conversation history
    conversation_context = ""
    if messages:
        conversation_context = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-5:] if isinstance(msg,(HumanMessage,AIMessage))])

    # Extract image data from the query, if present
    image_data = None
    user_query = messages[-1].content if messages else ""
    image_match = re.search(r"\[IMAGE:data:image/jpeg;base64,(.*?)\]", user_query)
    if image_match:
        image_data = image_match.group(1)
        user_query = re.sub(r"\[IMAGE:data:image/jpeg;base64,(.*?)\]", "[IMAGE]", user_query)  # Replace actual data with marker

    sys_inst = get_system_instruction(kid_info, ai_tone, conversation_context, user_name)
    if not messages or getattr(messages[0], "role", "").lower() != "system":
        messages.insert(0, SystemMessage(content=sys_inst))
        logger.info("System instruction injected")

     # Log the complete prompt structure
    prompt_data = {
        "system_prompt": sys_inst,
        "kids_information": kid_info,
        "ai_tone": ai_tone,
        "user_name": user_name,  # Added user name to logging
        "conversation_context": conversation_context,
        "user_query": user_query, # Use modified query
        "has_image": image_data is not None
    }
    bot_logger.log_prompt(prompt_data)

    model = ChatOpenAI(
        base_url="http://localhost:15205/v1",
        model_name="gemini-2.0-flash",
        temperature=0.5,
        api_key="324"
    )

    # Invoke the model with the modified query
    response = model.invoke(messages)
    logger.info("Agent returned direct response")
    return {"messages": [response], "requires_tool": False, "current_tool": None, "chat_id": chat_id}

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
        logger.info("Image generation request detected")
        return "image_tool"

    return "agent"
def process_image_tool(state: AgentState) -> AgentState:
    """Process image generation request."""
    messages = state["messages"]
    chat_id = state["chat_id"]
    last_ai_message = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    last_human_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if not last_ai_message or not last_human_message:
        logger.error("Could not find necessary messages for image generation")
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
    
    result_message = (
        "🖼️ تصویر شما آماده است!\n\n"
        "[مشاهده تصویر](%s)" % image_url
    )
    
    # Create tool call record
    tool_call = {
        "tool_name": "ImageGenerator",
        "tool_input": {"prompt": prompt, "model": model},
        "tool_result": image_url
    }
    tool_calls = state.get("tool_calls", [])
    tool_calls.append(tool_call)
    
    # Escape reserved MarkdownV2 characters in the result message
    new_message = AIMessage(
        content=safe_escape_markdown_v2(result_message),
        additional_kwargs={"is_image_response": True}
    )
    
    return {
        "messages": state["messages"] + [new_message],
        "tool_calls": tool_calls,
        "requires_tool": False,
        "current_tool": None,
        "chat_id": state["chat_id"]  # ...existing code...
    }

def optimize_memory(state: AgentState) -> AgentState:
    THRESHOLD = 5  # Summarize every 5 conversation messages
    # Count only human and AI messages (exclude system messages)
    conv_messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    
    # Run summarization only if count is a multiple of THRESHOLD
    if len(conv_messages) % THRESHOLD != 0:
        logger.info(f"Skipping summarization; conversation message count: {len(conv_messages)}")
        bot_logger.log_stage("Summarization", f"Skipping summarization; count: {len(conv_messages)}", category="process")
        return state
    
    # 1. Extract the conversation history
    conversation = "\n".join([f"{msg.type}: {msg.content}" for msg in conv_messages])
    
    # 2. Create a summarization prompt
    summary_prompt = f"Summarize the following conversation briefly, focusing on the key topics and requests:\n\n{conversation}"
    
    # 3. Use the summarization LLM to generate a summary
    summary_response = summarization_llm.invoke([HumanMessage(content=summary_prompt)])
    summary_text = summary_response.content
    logger.info(f"Conversation summary: {summary_text}")
    bot_logger.log_stage("Chat Summarization", f"Summary result: {summary_text}", category="ai")
    
    # 4. Build a new message list: a system message with summary + last THRESHOLD messages.
    new_messages = [SystemMessage(content=f"Conversation Summary: {summary_text}")]
    new_messages.extend(state["messages"][-THRESHOLD:])
    logger.info(f"Session history optimized. New session count: {len(new_messages)}")
    bot_logger.log_stage("Summarization Complete", f"New session message count: {len(new_messages)}", category="process")
    
    return {
        "messages": new_messages,
        "tool_calls": state.get("tool_calls", []),
        "requires_tool": state.get("requires_tool", False),
        "current_tool": state.get("current_tool", None),
        "chat_id": state["chat_id"],
    }
# ============================================================
#             LANGGRAPH WORKFLOW DEFINITION
# ============================================================
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("image_tool", process_image_tool)
# Add the memory optimization as a node
workflow.add_node("optimize_memory", optimize_memory)

# Add conditional routing
workflow.add_conditional_edges(
    "agent",
    route_tool,
    {
        "image_tool": "image_tool",
        "agent": "optimize_memory"  # Go to optimization after agent
    }
)

workflow.add_edge(START, "agent")
workflow.add_edge("image_tool", "optimize_memory") # Go to optimization after tool
workflow.add_edge("optimize_memory", END) # Go to END after optimization.


# ============================================================
#              USER MEMORY (Simplified)
# ============================================================
class UserMemory:
    """
    Simplified UserMemory class to manage per-user settings.  We rely on LangGraph's
    checkpointer for conversation history.
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._kid_info = ""
        self._ai_tone = DEFAULT_AI_TONE
        self._user_name = ""  # Add user name field
        self._thread_id = user_id  # Default thread ID is the same as user ID
        self._reset_count = 0      # Track how many times memory has been reset

        # Load settings from MongoDB if they exist
        if mongodb_client:
            user_settings = user_settings_collection.find_one({"user_id": self.user_id})
            if user_settings:
                self._kid_info = user_settings.get("kid_info", "")
                self._ai_tone = user_settings.get("ai_tone", DEFAULT_AI_TONE)
                self._user_name = user_settings.get("user_name", "")  # Load user name
                self._thread_id = user_settings.get("thread_id", self.user_id)
                self._reset_count = user_settings.get("reset_count", 0)

    def get_kid_info(self) -> str:
        return self._kid_info

    def set_kid_info(self, kid_info: str):
        self._kid_info = kid_info
        self._save_settings()

    def get_ai_tone(self) -> str:
        return self._ai_tone

    def set_ai_tone(self, ai_tone: str):
        self._ai_tone = ai_tone
        self._save_settings()
        
    def get_user_name(self) -> str:
        """Get user's name from memory"""
        return self._user_name
    
    def set_user_name(self, user_name: str):
        """Save user's name to memory"""
        self._user_name = user_name
        self._save_settings()
        
    def get_thread_id(self) -> str:
        """Get thread ID for LangGraph conversation history"""
        return self._thread_id
    
    def create_new_thread_id(self) -> str:
        """Generate new thread ID after memory reset"""
        self._reset_count += 1
        self._thread_id = f"{self.user_id}_{self._reset_count}"
        self._save_settings()
        return self._thread_id

    def _save_settings(self):
        """Saves the user's settings to MongoDB."""
        if mongodb_client:
            user_settings_collection.update_one(
                {"user_id": self.user_id},
                {"$set": {
                    "kid_info": self._kid_info, 
                    "ai_tone": self._ai_tone,
                    "user_name": self._user_name,  # Added user name
                    "thread_id": self._thread_id,
                    "reset_count": self._reset_count
                }},
                upsert=True  # Create if it doesn't exist
            )

#Global dictionary to store UserMemory instances.  In a production system,
# you might want to use a more robust caching mechanism.
user_memory_cache: Dict[str, UserMemory] = {}

def get_user_memory(user_id: str) -> UserMemory:
    """Retrieves or creates a UserMemory object for the given user ID."""
    if (user_id not in user_memory_cache):
        user_memory_cache[user_id] = UserMemory(user_id)
    return user_memory_cache[user_id]

# ============================================================
#             MAIN AGENT FUNCTION (run_agent)
# ============================================================
def run_agent(query, chat_id, message_id):
    # Import the telegram bot instance when needed
    from telegram_bot import bot  
    bot_logger.log_stage("Query", f"Received query: {query[:50]}...", category="user_query")
    start_time = time()
    user_id = str(chat_id)  # Use chat_id as user_id

    user_memory = get_user_memory(user_id)
    thread_id = user_memory.get_thread_id()  # Use thread_id from user memory
    bot_logger.log_stage("User Memory", 
                       f"Loaded memory for user {user_id} (thread: {thread_id})", 
                       category="memory")

    # ------------------- Send Loading Message -------------------
    loading_text_escaped = escape_markdown_v2(LOADING_MESSAGE_TEXT)
    loading_message = bot.send_message(chat_id, loading_text_escaped, parse_mode=MARKDOWN_V2_PARSE_MODE)
    loading_msg_id = loading_message.message_id
    bot_logger.log_stage("Loading", "Loading message sent to user.", category="process")

    # ------------------- LangGraph App Invocation -------------------
    bot_logger.log_stage("LangGraph Invoke", f"Invoking LangGraph app with thread {thread_id}.", category="ai")
    config = {"configurable": {"thread_id": thread_id}}  # Use thread_id for LangGraph config

    #Use checkpointer for conversation history
    with MongoDBSaver.from_conn_string(mongodb_uri, db_name="bluecare_db", collection_name="langgraph_checkpoints") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)

        inputs: AgentState = {
            "messages": [HumanMessage(content=query)],
            "tool_calls": [],
            "requires_tool": False,
            "current_tool": None,
            "chat_id": user_id  # Pass user_id in the state
        }

        try:
            # Use .stream() for a better interactive experience
            collected_content = ""
            full_response = ""
            for output in graph.stream(inputs, config=config):
                # Check if the output contains a message
                if "optimize_memory" in output and "messages" in output["optimize_memory"]:
                    messages = output["optimize_memory"]["messages"]
                    if messages and len(messages) > 0:
                        collected_content += messages[-1].content
                elif "agent" in output and "messages" in output["agent"]:
                    collected_content += output["agent"]["messages"][-1].content
                elif "image_tool" in output and "messages" in output["image_tool"]:
                    collected_content += output["image_tool"]["messages"][-1].content

                # Check if the collected content appears to be a complete message
                if collected_content.strip():
                    full_response = collected_content
                    collected_content = ""  # Reset for the next message

            # Use the full response
            ai_response_content = full_response

            # Add safety check to prevent empty responses
            if not ai_response_content:
                ai_response_content = "متاسفم، نتوانستم پاسخ مناسبی بیابم. لطفا دوباره تلاش کنید."

            # First refine the response (format bullets, etc.)
            refined_response = refine_ai_response(ai_response_content)
            
            # Then convert bold markdown to Telegram-compatible format
            telegram_formatted = convert_bold_to_telegram_format(refined_response)
            
            # Finally escape for MarkdownV2 while preserving links
            safe_response = safe_escape_markdown_v2(telegram_formatted)
            
            bot_logger.log_stage("Response", f"Generated response: {refined_response[:50]}...", category="ai_response")

            # ------------------- Send Final Response to User -------------------
            try:
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading_msg_id,
                    text=safe_response,   # Keep using MarkdownV2 with proper escaping
                    parse_mode=MARKDOWN_V2_PARSE_MODE,
                )
            except ApiTelegramException as e:
                bot.send_message(chat_id, safe_response, parse_mode=MARKDOWN_V2_PARSE_MODE)
            bot_logger.log_stage("Response Sent", "AI response sent to user successfully.", category="ai_response")

            bot_logger.log_memory_details(user_memory, user_id)
        except Exception as e:
            error_message = f"Error during LangGraph execution: {e}"
            logger.error(error_message)
            bot_logger.log_stage("Error", error_message, category="error")
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_msg_id,
                text="An error occurred. Please try again.",  # plain error message
                parse_mode=MARKDOWN_V2_PARSE_MODE
            )

    elapsed_time = time() - start_time
    bot_logger.log_stage("Process Complete", f"Query processed in {elapsed_time:.2f}s.", category="process")

# ============================================================
#                MAIN BOT POLLING
# ============================================================
if __name__ == "__main__":
    import threading
    # Start G4F Interference API Server
    try:
        from g4f.api import run_api
    except ImportError:
        logging.error("g4f.api module not found. Install the 'g4f' package.")
        run_api = None

    if run_api is not None:
        def start_interference_api():
            logging.info("Starting G4F Interference API server on http://localhost:15205/v1 ...")
            run_api(bind="0.0.0.0:15205")
        api_thread = threading.Thread(target=start_interference_api, daemon=True, name="G4F-API-Thread")
        api_thread.start()
    else:
        logging.warning("G4F API server not started due to missing module.")

    # Call the separate Telegram bot start function
    from telegram_bot import start_bot
    start_bot()