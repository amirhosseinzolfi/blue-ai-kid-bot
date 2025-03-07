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
else:
    logger.warning("MongoDB not available, will operate with in-memory storage only")

# ============================================================
#          TELEGRAM BOT IMPORTS & SETUP
# ============================================================
import telebot
from telebot.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeAllGroupChats,
)
from telebot.apihelper import ApiTelegramException

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_token_here")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

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
    model_name="gemini-2.0-flash",
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
    model_name="gemini-1.5-flash",  # Use a faster model
    temperature=0.3,  # Lower temperature for more concise summaries
    api_key="324"
)
logger.info("Summarization LLM initialized.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize a separate model for image-based queries
mm_model = ChatOpenAI(model="gpt-4o")

# ============================================================
#         PROMPT TEMPLATES & CONSTANTS
# ============================================================
# Global settings for user-specific configuration
setting_data = {}
STICKER_SETTING = None  # Sticker ID if needed


# ============================================================
#              BOT LOGGER CLASS DEFINITION
# ============================================================
# ------------------------------ Utility Functions ------------------------------
def refine_ai_response(text: str) -> str:
    """Cleans up and reformats the AI's output."""
    text = re.sub(r"\[Sticker:.*?\]", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)
    text = re.sub(r"__([^_]+)__", r"*\1*", text)
    text = re.sub(r"^####\s+(.*?)$", r"🔶 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^###\s+(.*?)$", r"⭐ \1", text, flags=re.MULTILINE)
    text = re.sub(r"^##\s+(.*?)$", r"🔷 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^#\s+(.*?)$", r"🟣 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^(?:\s*[-*]\s+)(.*?)$", r"🔹 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^(?:\s*\d+\.\s+)(.*?)$", r"🔹 \1", text, flags=re.MULTILINE)
    return text.strip()

def escape_markdown_v2(text: str) -> str:
    """Escapes markdown sensitive characters for Telegram."""
    return re.sub(r"([_*[\]()~`>#+\-=|{}.!])", r"\\\1", text)

# ...existing BotLogger definition removed...

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

     # Get the last 5 messages from the conversation history
    conversation_context = ""
    if messages:
        conversation_context = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-5:] if isinstance(msg,(HumanMessage,AIMessage))])



    sys_inst = get_system_instruction(kid_info, ai_tone, conversation_context)
    if not messages or getattr(messages[0], "role", "").lower() != "system":
        messages.insert(0, SystemMessage(content=sys_inst))
        logger.info("System instruction injected")

     # Log the complete prompt structure
    prompt_data = {
        "system_prompt": sys_inst,
        "kids_information": kid_info,
        "ai_tone": ai_tone,
        "conversation_context": conversation_context,
        "user_query": messages[-1].content if messages else ""
    }
    bot_logger.log_prompt(prompt_data)


    model = ChatOpenAI(
        base_url="http://localhost:15205/v1",
        model_name="gemini-2.0-flash",
        temperature=0.5,
        api_key="324"
    )

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
        "current_tool": None,
        "chat_id": chat_id #Keep chat ID
    }

def optimize_memory(state: AgentState) -> AgentState:
    THRESHOLD = 5  # Summarize every 5 conversation messages
    # Count only human and AI messages (exclude system messages)
    conv_messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    
    # Run summarization only if count is a multiple of THRESHOLD
    if len(conv_messages) % THRESHOLD != 0:
        logger.info(f"Skipping summarization; conversation message count: {len(conv_messages)}")
        return state
    
    # 1. Extract the conversation history
    conversation = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in conv_messages]
    )
    
    # 2. Create a summarization prompt
    summary_prompt = f"Summarize the following conversation briefly, focusing on the key topics and requests:\n\n{conversation}"
    
    # 3. Use the summarization LLM to generate a summary
    summary_response = summarization_llm.invoke([HumanMessage(content=summary_prompt)])
    summary_text = summary_response.content
    logger.info(f"Conversation summary: {summary_text}")
    
    # 4. Build a new message list: a system message with summary + last THRESHOLD messages.
    new_messages = [SystemMessage(content=f"Conversation Summary: {summary_text}")]
    new_messages.extend(state["messages"][-THRESHOLD:])
    logger.info(f"Session history optimized. New session count: {len(new_messages)}")
    
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

        # Load settings from MongoDB if they exist
        user_settings = user_settings_collection.find_one({"user_id": self.user_id})
        if user_settings:
            self._kid_info = user_settings.get("kid_info", "")
            self._ai_tone = user_settings.get("ai_tone", DEFAULT_AI_TONE)

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

    def _save_settings(self):
      """Saves the user's settings to MongoDB."""
      user_settings_collection.update_one(
          {"user_id": self.user_id},
          {"$set": {"kid_info": self._kid_info, "ai_tone": self._ai_tone}},
          upsert=True  # Create if it doesn't exist
      )

#Global dictionary to store UserMemory instances.  In a production system,
# you might want to use a more robust caching mechanism.
user_memory_cache: Dict[str, UserMemory] = {}

def get_user_memory(user_id: str) -> UserMemory:
    """Retrieves or creates a UserMemory object for the given user ID."""
    if user_id not in user_memory_cache:
        user_memory_cache[user_id] = UserMemory(user_id)
    return user_memory_cache[user_id]

# ============================================================
#             MAIN AGENT FUNCTION (run_agent)
# ============================================================
def run_agent(query, chat_id, message_id):
    """
    Processes a user query using LangGraph, with MongoDB checkpointer integration.
    """
    bot_logger.log_stage("Query", f"Processing: {query[:50]}...", category="user_query")
    start_time = time()
    user_id = str(chat_id)  # Use chat_id as user_id

    user_memory = get_user_memory(user_id)

    # ------------------- Send Loading Message -------------------
    loading_text_escaped = escape_markdown_v2(LOADING_MESSAGE_TEXT)
    loading_message = bot.send_message(chat_id, loading_text_escaped, parse_mode=MARKDOWN_V2_PARSE_MODE)
    loading_msg_id = loading_message.message_id
    bot_logger.log_stage("Loading Msg", "Sent loading message", category="general", detail=f"Msg ID: {loading_msg_id}")


    # ------------------- LangGraph App Invocation -------------------
    bot_logger.log_stage("LangGraph Invoke", "Invoking LangGraph app", category="ai")
    config = {"configurable": {"thread_id": str(chat_id)}}  # Use chat_id for thread

    #Use checkpointer for conversation history
    with MongoDBSaver.from_conn_string(mongodb_uri, db_name="bluecare_db", collection_name="langgraph_checkpoints") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)

        inputs: AgentState = {
            "messages": [HumanMessage(content=query)],
            "tool_calls": [],
            "requires_tool": False,
            "current_tool": None,
            "chat_id": str(chat_id)  # Pass chat_id in the state
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

            refined_response = refine_ai_response(ai_response_content)
            bot_logger.log_stage("Response", f"Generated: {refined_response[:50]}...", category="ai_response")

            # ------------------- Send Final Response to User -------------------
            try:
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading_msg_id,
                    text=escape_markdown_v2(refined_response),
                    parse_mode=MARKDOWN_V2_PARSE_MODE,
                )
            except ApiTelegramException as e:
              bot.send_message(chat_id, escape_markdown_v2(refined_response), parse_mode=MARKDOWN_V2_PARSE_MODE)
            bot_logger.log_stage("Response Sent", "AI response sent to user", category="ai_response")

            bot_logger.log_memory_details(user_memory,user_id) #Simplified call

        except Exception as e:
          error_message = f"Error during LangGraph execution: {e}"
          logger.error(error_message)
          bot.edit_message_text(chat_id=chat_id, message_id=loading_msg_id, text=escape_markdown_v2("An error occurred. Please try again."))



    elapsed_time = time() - start_time
    bot_logger.log_stage("Process Complete", "Query processing finished", category="process", detail=f"Time: {elapsed_time:.2f}s")




# ============================================================
#     TELEGRAM BOT COMMAND & CALLBACK HANDLERS
# ============================================================
@bot.message_handler(commands=["start"])
def start_handler(message):
    reply_text = (
        "🌟 **بلو؛ دستیار هوشمند تربیتی شما!** 🌟\n\n"
        "به سامانه جدید ما خوش آمدید. برای شروع گفتگو، یک پیام ارسال کنید."
    )
    bot.reply_to(message, escape_markdown_v2(reply_text), parse_mode=MARKDOWN_V2_PARSE_MODE)
    logger.info(f"/start command from user {message.chat.id}")

@bot.message_handler(commands=["help"])
def help_handler(message):
    reply_text = "📘 راهنمای بات بلو: برای کمک، پیام خود را ارسال کنید."
    bot.reply_to(message, reply_text)
    logger.info(f"/help command from user {message.chat.id}")

@bot.message_handler(commands=["setting"])
def setting_handler(message):
    chat_id = message.chat.id
    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton("🎈 اطلاعات کودک", callback_data="kid_info"),
        InlineKeyboardButton("💬 لحن هوشمند", callback_data="ai_tone"),
        InlineKeyboardButton("♻️ پاکسازی حافظه", callback_data="refresh_memory"),
    )
    bot.send_message(chat_id, "🔧 گزینه تنظیمات را انتخاب کنید:", reply_markup=markup)
    console.log(f"[bold blue]/setting command from user {chat_id}[/bold blue]")
    setting_data[chat_id] = {}
    if STICKER_SETTING:
        bot.send_sticker(chat_id, STICKER_SETTING)
        logger.info(f"Sent SETTING sticker to user {chat_id}")

@bot.callback_query_handler(func=lambda call: call.data in ["kid_info", "kid_info_add", "kid_info_replace", "ai_tone", "refresh_memory"])
def callback_query_handler(call):
    chat_id = call.message.chat.id
    action = call.data
    if action == "kid_info":
        handle_kid_info_callback(call)
    elif action in ["kid_info_add", "kid_info_replace"]:
        mode = action.split("_")[-1]
        message_text = (
            f"🌟 {'افزودن' if mode == 'add' else 'ثبت'} اطلاعات جدید\n\n"
            "لطفاً اطلاعات کودک خود را وارد کنید.\n"
            f"این اطلاعات به داده‌های قبلی {'اضافه خواهد شد' if mode == 'add' else 'جایگزین میشود'}."
        )
        bot.send_message(chat_id, message_text)
        setting_data[chat_id] = {"kid_info_pending": True, "kid_info_mode": mode}
        logger.info(f"Handled {action} callback for user {chat_id}")
    elif action == "ai_tone":
        handle_ai_tone_callback(call)
    elif action == "refresh_memory":
        handle_refresh_memory(call)
    bot.answer_callback_query(call.id)
    logger.info(f"Callback query handled for action: {action}, user: {chat_id}")

def handle_kid_info_callback(call):
    chat_id = call.message.chat.id
    user_memory = get_user_memory(str(chat_id))
    current_info = user_memory.get_kid_info() or "هیچ اطلاعاتی ثبت نشده است."
    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton("➕ افزودن اطلاعات", callback_data="kid_info_add"),
        InlineKeyboardButton("🔄 جایگزینی اطلاعات", callback_data="kid_info_replace"),
    )
    message_text = (
        "📋 اطلاعات فعلی کودک شما:\n" + current_info + "\n\n"
        "چه تغییری میخواهید ایجاد کنید؟"
    )
    bot.send_message(chat_id, message_text, reply_markup=markup)
    logger.info(f"Handled kid_info callback for user {chat_id}")

def handle_ai_tone_callback(call):
    chat_id = call.message.chat.id
    bot.send_message(chat_id, f"لحن مورد نظر خود را اکنون ارسال کنید: (لحن فعلی: {DEFAULT_AI_TONE})")
    console.log(f"[bold blue]درخواست تنظیم لحن هوشمند توسط کاربر {call.from_user.id}[/bold blue]")
    setting_data[chat_id]["ai_tone_pending"] = True
    logger.info(f"Handled ai_tone callback for user {chat_id}")

def handle_refresh_memory(call):
    chat_id = call.message.chat.id
    user_id = str(chat_id)

    #Clear UserMemory cache entry
    if user_id in user_memory_cache:
                del user_memory_cache[user_id]

    # Clear MongoDB checkpointer entries for this chat ID
    with MongoDBSaver.from_conn_string(mongodb_uri, db_name="bluecare_db", collection_name="langgraph_checkpoints") as checkpointer:
        checkpointer.delete(config={"configurable": {"thread_id": user_id}})

    bot.send_message(
        chat_id,
        escape_markdown_v2("حافظه بات ریست شد. تمام گفتگوها و حافظه‌های قبلی پاک شدند."),
        parse_mode=MARKDOWN_V2_PARSE_MODE,
    )
    logger.info(f"Refreshed all memory for user {user_id}")

@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_text_message(message):
    chat_id = message.chat.id
    user_id = str(chat_id)
    user_memory = get_user_memory(user_id)
    global KIDS_INFO, DEFAULT_AI_TONE
    if (chat_id in setting_data and setting_data[chat_id].get("kid_info_pending")):
        mode = setting_data[chat_id].get("kid_info_mode")
        kid_info = message.text
        prompt = f"خلاصه و بهینه سازی اطلاعات کودک:\n{kid_info}\n\nخلاصه:"
        summarized_info = info_llm.invoke(prompt).content
        if mode == "add" and user_memory.get_kid_info():
            summarized_info = info_llm.invoke(f"اطلاعات قبلی:\n{user_memory.get_kid_info()}\n\nاطلاعات جدید:\n{summarized_info}\n\nادغام و بهینه سازی:").content
        user_memory.set_kid_info(summarized_info)
        setting_data[chat_id]["kid_info_pending"] = False
        del setting_data[chat_id]["kid_info_mode"]
        bot.send_message(chat_id, f"✅ اطلاعات کودک ثبت شد:\n{summarized_info}")
        logger.info(f"Kid info updated for user {chat_id}")
    elif (chat_id in setting_data and setting_data[chat_id].get("ai_tone_pending")):
        ai_tone = message.text
        prompt = f"خلاصه و بهینه سازی لحن هوشمند:\n{ai_tone}\n\nخلاصه:"
        summarized_tone = info_llm.invoke(prompt).content
        user_memory.set_ai_tone(summarized_tone)  # Update user-specific tone
        setting_data[chat_id]["ai_tone_pending"] = False
        bot.send_message(chat_id, f"✅ لحن هوشمند ثبت شد:\n{summarized_tone}")
        logger.info(f"AI tone updated for user {chat_id}")
    else:
        # config = {"configurable": {"thread_id": str(chat_id), "user_id": user_id}} # User ID no longer needed here
        run_agent(message.text, chat_id=chat_id, message_id=message.message_id)
        logger.info(f"Handled text message for user {chat_id}")

@bot.message_handler(content_types=["photo"])
def handle_photo_message(message):
    chat_id = message.chat.id
    # Retrieve the photo file and download it
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    # Convert image to base64
    image_data = base64.b64encode(downloaded_file).decode("utf-8")
    # Prepare a multimodal message
    mm_content = [
        {"type": "text", "text": "Describe what you see in this image"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ]
    mm_message = HumanMessage(content=mm_content)
    # Invoke the model
    response = mm_model.invoke([mm_message])
    # Send the AI response back to the user
    bot.send_message(chat_id, refine_ai_response(response.content))

# ============================================================
#         TEXT MESSAGE HANDLER & BOT COMMAND SETUP
# ============================================================
def setup_bot_commands():
    """Sets up the Telegram bot commands for both private and group chats."""
    bot_commands = [
        BotCommand("start", "شروع"),
        BotCommand("help", "راهنما"),
        BotCommand("setting", "تنظیمات"),
    ]
    try:
        bot.set_my_commands(commands=bot_commands, scope=BotCommandScopeDefault())
        bot.set_my_commands(commands=bot_commands, scope=BotCommandScopeAllGroupChats())
        logger.info("Bot commands setup completed.")
    except Exception as e:
        logger.error(f"Error setting up bot commands: {e}")


# ============================================================
#                MAIN BOT POLLING
# ============================================================
if __name__ == "__main__":
    setup_bot_commands()
    bot.polling(non_stop=True)