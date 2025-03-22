import os
import base64
import logging
import re
from telebot import TeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, BotCommand, BotCommandScopeDefault, BotCommandScopeAllGroupChats
from telebot.apihelper import ApiTelegramException

# Import necessary helper functions and constants from bot2.py
from bot2 import run_agent, escape_markdown_v2, DEFAULT_AI_TONE, KIDS_INFO, LOADING_MESSAGE_TEXT
from bot2 import get_user_memory  # and any other required functions
from bot2 import info_llm  # for summaries in settings

# Import SystemMessage and HumanMessage for use with info_llm
from bot2 import SystemMessage, HumanMessage
from prompts import get_info_prompt, get_ai_tone_prompt

# Initialize Telegram Bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_token_here")
bot = TeleBot(TELEGRAM_BOT_TOKEN)
bot_info = bot.get_me()
bot_username = bot_info.username if bot_info else ""

# Global variable for temporary settings
setting_data = {}
STICKER_SETTING = None  # Sticker ID if needed
# Add this import at the top of your telegram_bot.py file
from prompts import story_prompt
# Add this import if not already present
from prompts import story_prompt, luluby_prompt

@bot.message_handler(commands=["lullabies"])
def handle_lullabies(message):
    """
    Handle the request for night lullabies.
    When this command is triggered, it sends the predefined luluby prompt to the AI.
    """
    chat_id = message.chat.id
    message_id = message.message_id
    user_id = str(message.from_user.id)
    
    # Log the lullaby request
    logging.info(f"Lullaby request from user {user_id}")
    
    # Use the luluby prompt from prompts.py instead of user input
    from bot2 import run_agent
    
    # Run the agent with the luluby prompt instead of user message text
    run_agent(luluby_prompt, chat_id, message_id)
# Add this command handler in your bot setup section
@bot.message_handler(commands=["story"])
def handle_educational_stories(message):
    """
    Handle the request for educational stories.
    When this command is triggered, it sends the predefined story prompt to the AI.
    """
    chat_id = message.chat.id
    message_id = message.message_id
    user_id = str(message.from_user.id)
    
    # Log the educational story request
    logging.info(f"Educational story request from user {user_id}")
    
    # Use the story prompt from prompts.py instead of user input
    from bot2 import run_agent
    
    # Run the agent with the story prompt instead of user message text
    run_agent(story_prompt, chat_id, message_id)
# ================================
# Telegram Command and Callback Handlers
# ================================
@bot.message_handler(commands=["start"])
def start_handler(message):
    chat_id = message.chat.id
    user_id = str(chat_id)
    user_name = message.from_user.first_name if message.from_user else ""
    if user_name:
        from bot2 import get_user_memory
        user_memory = get_user_memory(user_id)
        user_memory.set_user_name(user_name)
        logging.info(f"Saved user name '{user_name}' for user {user_id}")
    reply_text = (
        f"🌟 بلو؛ دستیار هوشمند تربیتی شما! 🌟\n\n"
        "آیا در مسیر تربیت فرزندتان با چالش‌هایی مثل لجبازی، وابستگی به گوشی، یا کمبود ایده‌های آموزشی مواجهید؟  \n"
        "بلو اینجاست تا با راهکارهای هوشمند، مسیر تربیت رو برای شما هموار و هیجان‌انگیز کنه!\n\n"
        "✨ ویژگی‌های منحصربه‌فرد بلو:  \n"
        "🔹 تست شخصیت کودک: شناخت علمی عمیق برای تربیتی بهینه  \n"
        "🔹 بازی‌های آموزشی: جایگزینی جذاب برای فضای مجازی  \n"
        "🔹 قصه‌های آموزنده: انتقال ارزش‌های اخلاقی و مهارتی  \n"
        "🔹 نکات تربیتی و کمک درسی: راهکارهای کاربردی و موثر  \n"
        "🔹 کاردستی‌هيالای خلاقانه و تصاویر شخصی‌شده: تقویت خلاقیت و تمرکز\n\n"
        "💥 همین حالا به جمع خانواده بلو بپیوندید و تجربه‌ای نوین از تربیت دیجیتال داشته باشید! 🚀\n"
    )
    bot.reply_to(message, escape_markdown_v2(reply_text), parse_mode="MarkdownV2")
    logging.info(f"/start command from user {chat_id}")

@bot.message_handler(commands=["help"])
def help_handler(message):
    reply_text = "📘 راهنمای بات بلو: برای کمک، پیام خود را ارسال کنید."
    bot.reply_to(message, reply_text)
    logging.info(f"/help command from user {message.chat.id}")

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
    logging.info(f"/setting command from user {chat_id}")
    setting_data[chat_id] = {}
    if STICKER_SETTING:
        bot.send_sticker(chat_id, STICKER_SETTING)
        logging.info(f"Sent SETTING sticker to user {chat_id}")

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
        logging.info(f"Handled {action} callback for user {chat_id}")
    elif action == "ai_tone":
        handle_ai_tone_callback(call)
    elif action == "refresh_memory":
        handle_refresh_memory(call)
    bot.answer_callback_query(call.id)
    logging.info(f"Callback query handled for action: {action}, user: {chat_id}")

def handle_kid_info_callback(call):
    chat_id = call.message.chat.id
    from bot2 import get_user_memory
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
    logging.info(f"Handled kid_info callback for user {chat_id}")

def handle_ai_tone_callback(call):
    chat_id = call.message.chat.id
    bot.send_message(chat_id, f"لحن مورد نظر خود را اکنون ارسال کنید: (لحن فعلی: {DEFAULT_AI_TONE})")
    logging.info(f"Handled ai_tone callback for user {chat_id}")
    setting_data[chat_id]["ai_tone_pending"] = True

def handle_refresh_memory(call):
    chat_id = call.message.chat.id
    from bot2 import get_user_memory, DEFAULT_AI_TONE
    user_id = str(chat_id)
    logging.info(f"Starting memory reset for user {user_id}")
    try:
        user_memory = get_user_memory(user_id)
        old_thread_id = user_memory.get_thread_id()
        new_thread_id = user_memory.create_new_thread_id()
        logging.info(f"Generated new thread ID: {new_thread_id} (was: {old_thread_id})")
        user_memory._kid_info = ""
        user_memory._ai_tone = DEFAULT_AI_TONE
        user_memory._save_settings()
        # Clear MongoDB checkpoints if applicable (using bot2's logic)
        from pymongo import MongoClient
        # ...existing code for clearing checkpoints...
        bot.send_message(
            chat_id,
            escape_markdown_v2("✅ حافظه بات ریست شد. تمام گفتگوها، تنظیمات و حافظه‌های قبلی پاک شدند."),
            parse_mode="MarkdownV2",
        )
        logging.info(f"Memory reset completed for user {user_id}")
    except Exception as e:
        logging.error(f"Error during memory reset: {str(e)}")
        bot.send_message(chat_id, escape_markdown_v2("❌ خطا در پاکسازی حافظه. لطفاً دوباره تلاش کنید."))

@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_text_message(message):
    # For group chats, respond only if:
    # 1. The message is a reply to a bot message, OR
    # 2. The message contains "tara" or "تارا" or tags the bot.
    if message.chat.type in ["group", "supergroup"]:
        reply_condition = (
            message.reply_to_message is not None and 
            message.reply_to_message.from_user is not None and 
            message.reply_to_message.from_user.id == bot_info.id
        )
        text_lower = message.text.lower() if message.text else ""
        tag_condition = bot_username and f"@{bot_username}".lower() in text_lower
        if not (reply_condition or re.search(r'\b(tara|تارا)\b', text_lower) or tag_condition):
            return
    chat_id = message.chat.id
    from bot2 import get_user_memory, info_llm
    user_id = str(chat_id)
    user_name = message.from_user.first_name if message.from_user else ""
    user_memory = get_user_memory(user_id)
    if user_name and not user_memory.get_user_name():
        user_memory.set_user_name(user_name)
        logging.info(f"Saved user name '{user_name}' for user {user_id}")
    global KIDS_INFO, DEFAULT_AI_TONE
    if (chat_id in setting_data and setting_data[chat_id].get("kid_info_pending")):
        mode = setting_data[chat_id].get("kid_info_mode")
        kid_info = message.text
        
        # Use get_info_prompt with the kid_info as system instruction
        system_instruction = get_info_prompt(kid_info)
        summarized_info = info_llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content="organize this information for child profile")
        ]).content
        
        if mode == "add" and user_memory.get_kid_info():
            merged_info = f"اطلاعات قبلی:\n{user_memory.get_kid_info()}\n\nاطلاعات جدید:\n{summarized_info}"
            # Use get_info_prompt again for the merged info
            system_instruction = get_info_prompt(merged_info)
            summarized_info = info_llm.invoke([
                SystemMessage(content=system_instruction),
                HumanMessage(content="Merge these pieces of information into a cohesive child profile")
            ]).content
            
        user_memory.set_kid_info(summarized_info)
        setting_data[chat_id]["kid_info_pending"] = False
        del setting_data[chat_id]["kid_info_mode"]
        bot.send_message(chat_id, f"✅ اطلاعات کودک ثبت شد:\n{summarized_info}")
        logging.info(f"Kid info updated for user {chat_id}")
    
    elif (chat_id in setting_data and setting_data[chat_id].get("ai_tone_pending")):
        ai_tone = message.text
        # For tone, we'll use a more generic approach with get_ai_tone_prompt
        from prompts import get_ai_tone_prompt
        prompt = get_ai_tone_prompt(ai_tone)
        summarized_tone = info_llm.invoke(prompt).content
        user_memory.set_ai_tone(summarized_tone)
        setting_data[chat_id]["ai_tone_pending"] = False
        bot.send_message(chat_id, f"✅ لحن هوشمند ثبت شد:\n{summarized_tone}")
        logging.info(f"AI tone updated for user {chat_id}")
    
    else:
        run_agent(message.text, chat_id=chat_id, message_id=message.message_id)
        logging.info(f"Handled text message for user {chat_id}")

@bot.message_handler(content_types=["photo"])
def handle_photo_message(message):
    chat_id = message.chat.id
    from bot2 import mm_model
    user_id = str(chat_id)
    user_name = message.from_user.first_name if message.from_user else ""
    from bot2 import get_user_memory
    user_memory = get_user_memory(user_id)
    if user_name and not user_memory.get_user_name():
        user_memory.set_user_name(user_name)
    loading_message = bot.send_message(chat_id, escape_markdown_v2(LOADING_MESSAGE_TEXT), parse_mode="MarkdownV2")
    loading_msg_id = loading_message.message_id
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        image_data = base64.b64encode(downloaded_file).decode("utf-8")
        mm_content = [
            {"type": "text", "text": "Describe what you see in this image in detail, focusing on elements that might be relevant to child care or parenting. Respond in Persian (Farsi)."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
        from bot2 import HumanMessage
        mm_message = HumanMessage(content=mm_content)
        logging.info(f"Processing image from user {chat_id}")
        response = mm_model.invoke([mm_message])
        refined_response = response.content  # or use a refine function if desired
        try:
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_msg_id,
                text=escape_markdown_v2(refined_response),
                parse_mode="MarkdownV2",
            )
        except ApiTelegramException as e:
            bot.send_message(chat_id, escape_markdown_v2(refined_response), parse_mode="MarkdownV2")
        logging.info(f"Image analysis completed for user {chat_id}")
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        try:
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_msg_id,
                text=escape_markdown_v2("متاسفانه در پردازش تصویر خطایی رخ داد. لطفا دوباره تلاش کنید."),
                parse_mode="MarkdownV2"
            )
        except:
            bot.send_message(
                chat_id, 
                escape_markdown_v2("متاسفانه در پردازش تصویر خطایی رخ داد. لطفا دوباره تلاش کنید."),
                parse_mode="MarkdownV2"
            )

def setup_bot_commands():
    bot_commands = [
        BotCommand("start", "شروع"),
        BotCommand("help", "راهنما"),
        BotCommand("setting", "تنظیمات"),
        BotCommand("story","قصه_های_آموزنده"),
        BotCommand("lullabies", "لالایی های شبانه برای کودکان"),
    ]
    try:
        bot.set_my_commands(commands=bot_commands, scope=BotCommandScopeDefault())
        bot.set_my_commands(commands=bot_commands, scope=BotCommandScopeAllGroupChats())
        logging.info("Bot commands setup completed.")
    except Exception as e:
        logging.error(f"Error setting up bot commands: {e}")
def start_bot():
    setup_bot_commands()
    logging.info("Bot commands configured. Starting polling...")
    bot.polling(non_stop=True)

# Allow other modules (e.g. bot2.py) to import bot
if __name__ == "__main__":
    start_bot()
