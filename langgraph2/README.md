# langgraph2 Project

## Overview
This project is a Telegram bot designed to assist parents with child care advice. It utilizes an AI model to provide personalized responses based on user input and maintains user-specific settings and memory.

## Project Structure
```
langgraph2
├── src
│   ├── bot2.py          # Main logic for the Telegram bot
├── .gitignore           # Files and directories to ignore by Git
├── requirements.txt     # Python dependencies for the project
└── README.md            # Documentation for the project
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <your-github-repo-url>
   cd langgraph2
   ```

2. **Install Dependencies**
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Bot**
   Execute the bot script:
   ```bash
   python src/bot2.py
   ```

## Usage
- Start the bot by sending the `/start` command in your Telegram chat.
- Use the `/help` command to get assistance with the bot's features.
- Access settings using the `/setting` command to manage child information and AI tone.

## Contributing
Feel free to fork the repository and submit pull requests for any improvements or bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.