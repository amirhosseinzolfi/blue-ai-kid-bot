DEFAULT_AI_TONE = "دوستانه"
KIDS_INFO = ""  # Added default kid info as an empty string
LOADING_MESSAGE_TEXT = "درحال فکر کردن 🧐..."
MARKDOWN_V2_PARSE_MODE = "MarkdownV2"
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage

# System instruction prompt for Blue
OPTIMIZED_SYSTEM_PROMPT = (
    "You are a digital kids nurturing assistant named Blue. Provide accurate, personalized, and natural advice to parents regarding child care. "
    "Consider the following child information: {kids_information}. "
    "Also, take into account the previous conversation history: {conversation_context}. "
    "Relevant long-term memories: {long_term_memory}. "
    "Adjust your tone based on this setting: {ai_tone}. "
    "Do not include any greetings, salutations, or sticker instructions in your response. "
    "Be direct, concise, and fully answer the user's query in warm, supportive Persian. "
    "If the user asks to generate an image or visual content, identify this as an image generation request and indicate that you'll use the ImageGenerator tool. "
    "Example: 'Create an image of a cat' -> Use ImageGenerator."
)

system_prompt_template = PromptTemplate.from_template(OPTIMIZED_SYSTEM_PROMPT)

def get_system_instruction(kid_info: str, ai_tone: str, conversation_context: str, long_term_memory: str = "") -> str:
    """Returns the system instruction with variables replaced."""
    return OPTIMIZED_SYSTEM_PROMPT.format(
        kids_information=kid_info,
        conversation_context=conversation_context,
        long_term_memory=long_term_memory,
        ai_tone=ai_tone
    )

def get_system_message(kid_info: str, ai_tone: str, conversation_context: str, long_term_memory: str = "") -> SystemMessage:
    """Returns a SystemMessage object with the filled system instruction."""
    instruction = get_system_instruction(kid_info, ai_tone, conversation_context, long_term_memory)
    return SystemMessage(content=instruction)

def get_summary_prompt(conversation: str) -> str:
    """Returns the prompt to ask the LLM to summarize a conversation."""
    return f"Summarize the following conversation briefly, focusing on the key topics and requests:\n\n{conversation}"

def get_image_prompt(prompt_text: str, model: str = "midjourney") -> str:
    """Returns a prompt to instruct the LLM regarding image generation."""
    return f"Generate an image using the {model} model with the following prompt:\n{prompt_text}"

def get_info_prompt(info_text: str) -> str:
    """Returns a prompt to summarize or optimize child information."""
    return f"Summarize and optimize the following child information:\n{info_text}\n\nSummary:"

def get_ai_tone_prompt(tone_text: str) -> str:
    """Returns a prompt to summarize or optimize the AI tone."""
    return f"Summarize and optimize the following AI tone description:\n{tone_text}\n\nSummary:"
