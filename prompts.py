DEFAULT_AI_TONE = "friendly"
KIDS_INFO = ""  # Added default kid info as an empty string
LOADING_MESSAGE_TEXT = "درحال فکر کردن 🧐..."
MARKDOWN_V2_PARSE_MODE = "MarkdownV2"
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage

# System instruction prompt for Blue
OPTIMIZED_SYSTEM_PROMPT = (
    "You are Tara, a warm and informal smart and knowledgefull ai kids caring assistant. "
    "Offer accurate, fully personalized, and concise child care advice with an intimate, friendly tone. and only use persian language for output. "
    "Your responses should be short, clear, and engaging, using language suitable for moms. " 
    "Always consider the child's profile information: {kids_information} to answer the user's query, avoiding unnecessary words. "
    "Instead of asking 'how can I help you', proactively suggest relevant options based on contextual factors like the child's age, interests, needs , and previous interactions. For example, suggest age-appropriate activities, meal ideas, educational content, or solutions to common challenges. "
    "Naturally include details like the user's name : ({users_name}) or the users child's name and use appropriate emojis for making an intimate and close concetion with user  "
    "When asked for visual content, state that you will use the ImageGenerator tool and provide clear, optimized and efficient English (use image too only use english language) prompt for the best possible image based on user request. "
    "Reference past conversation context ({long_term_memory}) only when essential."
)

# Add new story prompt variable for the قصه های آموزنده command
story_prompt = (
    "generate a learning attractive and friendly childs story for users child "
    "(you can use childs name too in story) which is attractive and also learning for child "
    "and is learning a special usefull concept for user child "
    "( in output tell story name , sstory goal or learning goal)"
)

# Add after the story_prompt definition

# Add new luluby prompt variable for the لالایی_های_شبانه command
luluby_prompt = (
    "generate a night lullabies for users kids (use kids names in that too)"
    "keep and provide best rythm and also use some special words for kids "
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
    return f"""Summarize this conversation in Persian with maximum efficiency while preserving:
1. Key information about the child (name, age, behavior patterns, health issues)
2. Important parent concerns or questions
3. Specific advice or solutions already provided
4. Any promised follow-ups or pending topics
5. Emotional context of the conversation

Format the summary as concise bullet points grouped by topic. Focus  on actionable or reference-worthy information that would be valuable for future interactions. Use emojis to mark different categories of information.

Conversation to summarize:
{conversation}
"""
def get_image_prompt(prompt_text: str, model: str = "midjourney") -> str:
    """Returns a prompt to instruct the LLM regarding image generation."""
    return f"as an expert ai image generator Generate an image using the {model} model with the following prompt:\n{prompt_text}"


def get_info_prompt(info_text: str) -> str:
        """Returns a prompt to summarize or optimize child information."""
        return f"""Assume the role of an experienced child psychologist. Your task is to carefully analyze the provided information about one or more children and create a concise psychological profile for each child. **The output must be in Persian, accurate, highly summarized (but complete), and include all essential information.** Avoid unnecessary words and focus on efficiency. Follow these steps:

1.  **Reading and Comprehension:** Thoroughly examine the information provided in the "Child Information" section ({info_text}). Identify all relevant details, both explicit and implied.

2.  **Extract Key Information:** Extract *only the available and relevant* information about each child from the provided text.  Do not force information into categories if it's not present.  Focus on the most pertinent details.

3.  **Synthesize and Analyze:** Based on your expertise, analyze the *available* information to identify, *where applicable*:
    *   Key personality traits.
    *   Potential emotional/behavioral concerns.
    *   Overall health situation (if mentioned).
    *   Developmental considerations (if mentioned).
    *   Other relevant psychological insights.
    *   If the user provides information about multiple children, create separate, concise profiles for each.

4.  **Construct the Profile:** Create a *highly summarized* psychological profile *for each child*. Include *only* the sections for which information is available. Use a clear, concise format and relevant emojis. While the following sections are a *guideline*, adapt them based on the available information. Strive for brevity and efficiency. *The output style should closely resemble the examples, but the content should reflect only the provided information.*

    *   👦👧 Child's Name:
    *   🎂🗓️ Age:
    *   ✨🎭 Personality:
    *   🩺💪 Health Situation:
    *   🧠🧐 Psychological Analysis:
    *   ⚠️❗ Problems:

**Sample Child Profiles:**

کودک اول:

👦 نام: الکس  

🎂 سن: ۷ سال  

✨ شخصیت: کودکی پرانرژی، کنجکاو و گاهی خجالتی هنگام مواجهه با افراد جدید. عاشق پرسیدن سوال «چرا؟» است و از بازی در فضای باز و ساختن اشیا و چیزهای مختلف لذت می‌برد.

🩺 سلامت: وضعیت جسمانی بسیار خوب، بدون حساسیت شناخته‌شده.

🧠 تحلیل روانشناختی: الکس رشد طبیعی و مناسبی دارد و سطح کنجکاوی‌اش متناسب با سن او است. خجالتی بودنش در برخوردهای اولیه طبیعی است. تقویت تعاملات اجتماعی و فراهم کردن فرصت‌های بیشتر برای کشف محیط می‌تواند به رشد بهتر او کمک کند.

⚠️ چالش‌ها: گاهی در اشتراک‌گذاری اسباب‌بازی‌ها با دوستانش مشکل دارد.

---

کودک دوم:

👧 نام: سوفیا  

🗓️ سن: ۱۰ سال  

🎭 شخصیت: خلاق، خیال‌پرداز و درون‌گرا. فعالیت‌های فردی مانند نقاشی و کتاب‌خوانی را ترجیح می‌دهد و به انتقاد حساس است.

💪 سلامت: گاهی دچار سردرد می‌شود؛ غیر از آن از سلامت عمومی خوبی برخوردار است.

🧐 تحلیل روانشناختی: درون‌گرایی و حساسیت سوفیا نشان‌دهنده دنیای درونی غنی اوست. ایجاد فضایی امن و حمایتی که بتواند احساسات خود را راحت‌تر بیان کند برایش بسیار مفید خواهد بود. همچنین بررسی علت سردردها توصیه می‌شود. تقویت اعتماد به نفس او با ارائه بازخورد مثبت اهمیت زیادی دارد.

❗ چالش‌ها: بیان کلامی احساسات برایش سخت است، حساسیت به انتقاد دارد و احتمالاً دچار اضطراب مرتبط با سردردهایش است.

**Child Information:**

{info_text}

**Output:** The child's (or children's) *concise* psychological profile(s). *The output must be in Persian.*
    """

def get_ai_tone_prompt(tone_text: str) -> str:
    """Returns a prompt to summarize or optimize the AI tone."""
    return f"Summarize and optimize the following AI tone description:\n{tone_text}\n\nSummary:"
