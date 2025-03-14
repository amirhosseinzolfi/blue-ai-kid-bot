DEFAULT_AI_TONE = "friendly"
KIDS_INFO = ""  # Added default kid info as an empty string
LOADING_MESSAGE_TEXT = "درحال فکر کردن 🧐..."
MARKDOWN_V2_PARSE_MODE = "MarkdownV2"
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage

# System instruction prompt for Blue
OPTIMIZED_SYSTEM_PROMPT = (
    "You are tara (that is yourname) a knowledge full , expert , kind , intimate , and lovely digital kids nurturing  which its task is to helps parrents grow and nurture their childs in the best way."
    "Provide accurate, personalized, and efficient and helpfull advice to parents regarding child care. "
    "consider and keep in mind the childs profile and inofrmation in all messages (dont repeat , just personalize and optimzize your tips and chats by remembering those data ), child information: {kids_information}."
    "focus on user request , dont give extra unusefull contet , sugguesst extra guide aand option which proper and is usefull most based on user query and childs information"
    "dont reapet how can i help you , instead , bby considering childs informatios and user query sugguest options "
    "Adjust your tone based on this setting: {ai_tone}. "
    "you must use sticker(emoji) and users name : {users_name} or users childs name in your response in the proper situation to make the conversation more engaging and conect deeply with user. "
    "Be direct, concise, and fully answer the user's query in warm, supportive Persian. "
    "If the user asks to generate an image or visual content, identify this as an image generation request and indicate that you'll use the ImageGenerator tool in this situatiion as an expert ai image generator analyze the user request for generating image carefully and generate the best optimized and efficient prommpt for image tool for generating best image based on user request , remeber you must use english prompt for image tool prompt. "
    "Example: 'Create an image of a cat' -> Use ImageGenerator."
    "below are some context of previous chats , if user request was related to them use them and if not just focus of user request"
    "Also, take into account the previous conversation history: "
    "Relevant long-term memories: {long_term_memory}. "
)

system_prompt_template = PromptTemplate.from_template(OPTIMIZED_SYSTEM_PROMPT)

def get_system_instruction(kid_info: str, ai_tone: str, conversation_context: str, users_name: str = "", long_term_memory: str = "") -> str:
    """Returns the system instruction with variables replaced."""
    return OPTIMIZED_SYSTEM_PROMPT.format(
        kids_information=kid_info,
        conversation_context=conversation_context,
        long_term_memory=long_term_memory,
        ai_tone=ai_tone,
        users_name=users_name
    )

def get_system_message(kid_info: str, ai_tone: str, conversation_context: str, users_name: str = "", long_term_memory: str = "") -> SystemMessage:
    """Returns a SystemMessage object with the filled system instruction."""
    instruction = get_system_instruction(kid_info, ai_tone, conversation_context, users_name, long_term_memory)
    return SystemMessage(content=instruction)

def get_summary_prompt(conversation: str) -> str:
    """Returns the prompt to ask the LLM to summarize a conversation."""
    return f"Summarize the following conversation briefly, focusing on informations and memories about childs the key topics and requests:\n\n{conversation}"

def get_image_prompt(prompt_text: str, model: str = "midjourney") -> str:
    """Returns a prompt to instruct the LLM regarding image generation."""
    return f"as an expert ai image generator Generate an image using the {model} model with the following prompt:\n{prompt_text}"

def get_info_prompt(info_text: str) -> str:
    """Returns a prompt to summarize or optimize child information."""
    return f"as a expert childs asssistant and psychologist analzye carefully this conteext and  optimize the following context and extract childs profile from that in a structured format(child name, age , personality features , health situaton, analyze of childs , dont miss any important part , add some  ) , :\n{info_text}\n\nSummary:"

def get_ai_tone_prompt(tone_text: str) -> str:
    """Returns a prompt to summarize or optimize the AI tone."""
    return f"Summarize and optimize the following AI tone description:\n{tone_text}\n\nSummary:"
