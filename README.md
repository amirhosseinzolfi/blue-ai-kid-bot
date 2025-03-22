# Blue AI Kid Bot 🧒🤖

A specialized Telegram bot built to interact with children in a safe, educational, and engaging way. The bot uses LangChain, LangGraph, and various AI models to provide kid-friendly responses, generate images, and maintain contextual conversations.

## 🌟 Features

- **Kid-friendly AI Conversations**: Safe, age-appropriate responses for children
- **Customizable AI Personality**: Adjust the tone and style of the AI for different age groups
- **Image Generation**: Create illustrations based on children's requests
- **Persistent Memory**: Remember user preferences and conversation context
- **Multi-modal Support**: Process both text and images for richer interactions
- **Conversation Summarization**: Automatically optimize memory usage by summarizing longer conversations

## 🛠️ System Requirements

- Python 3.9+
- MongoDB (for persistent storage)
- Internet connection for API calls
- Sufficient memory for language model operation (minimum 8GB RAM recommended)

## 📋 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/blue-ai-kid-bot.git
   cd blue-ai-kid-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up MongoDB (if not already running):
   ```bash
   # Install MongoDB community edition based on your OS
   # Start MongoDB service
   sudo systemctl start mongod
   ```

## ⚙️ Configuration

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather) and get your bot token

2. Set environment variables (you'll be prompted for these when first running the bot):
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
   ```

3. Optional: Customize default settings in the `prompts.py` file for:
   - Default AI tone
   - Kid information templates
   - System instructions

## 🚀 Usage

Start the bot:
```bash
python bot2.py
```

The bot will:
1. Start a local G4F API server on port 15205
2. Connect to MongoDB
3. Begin polling for Telegram messages

### Bot Commands

- `/start` - Begin interacting with the bot
- `/reset` - Reset conversation memory
- `/name [name]` - Set your name for personalized interaction
- `/kid [information]` - Set information about the kid (age, interests, etc.)
- `/tone [tone]` - Set the AI's tone (e.g., "friendly teacher", "silly friend")

## 🏗️ Architecture

The bot uses a sophisticated architecture:

- **LangGraph Workflow**: A state machine that handles message routing, tool selection, and memory optimization
- **MongoDB Storage**: For persistent user preferences and conversation history
- **Rich Logging**: Detailed console output with structured, colorized information
- **Multi-Model Setup**: Different models for general conversation, summarization, and image generation

## 📊 Components

- `bot2.py`: Main application with LangGraph workflow and core logic
- `bot_logger.py`: Rich console logging for development and debugging
- `telegram_bot.py`: Telegram interface and command handling
- `prompts.py`: Template strings for AI system instructions and prompts

## 🔐 Security Notes

- The bot is designed with child safety in mind
- AI responses are filtered for age-appropriate content
- No personal data is shared with external services beyond what's necessary for the AI functions
- User data is stored in your own MongoDB instance, not shared with third parties

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support, please open an issue on the GitHub repository or contact the maintainers directly.
