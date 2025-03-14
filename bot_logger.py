from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from time import time
import logging
from datetime import datetime  # Added for timestamp info

logger = logging.getLogger(__name__)

class BotLogger:
    """Structured logger using Rich for visual terminal output."""
    def __init__(self):
        self.console = Console()
        self.start_time = time()
        self.stage_counter = 0
        self.log_styles = {
            "memory": "green",
            "ai": "blue",
            "process": "magenta",
            "error": "red",
            "general": "grey70",
            "user_query": "cyan",
            "ai_response": "yellow",
            "memory_details": "light_slate_blue",
            "prompt": "bright_blue"
        }
        logger.info("BotLogger initialized.")

    def log_stage(self, stage: str, message: str, category: str = "general", detail: str = None):
        self.stage_counter += 1
        # Added current timestamp in hh:mm:ss format
        timestamp = datetime.now().strftime("%H:%M:%S")
        header = f"[{timestamp}] {stage}"
        style = self.log_styles.get(category, "default")
        formatted_msg = f"[{style}]{message}[/{style}]"
        if detail:
            formatted_msg += f"\n  [grey70]{detail}[/grey70]"
        panel = Panel(
            formatted_msg,
            title=header,
            title_align="left",
            border_style=style,
            padding=(0, 1),
            expand=False,
        )
        self.console.print(panel)

    def log_info(self, message: str):
        # New minimal info log for quick one-line messages
        self.console.print(f"[green][INFO][/green] {message}")

    def log_error(self, error: str):
        panel = Panel(f"[bold red]{error}[/bold red]", title="ERROR", border_style="red", expand=False)
        self.console.print(panel)
        logger.error(error)

    def log_memory_details(self, user_memory, user_id):
        try:
            table = Table(title=f"Memory Details for User: {user_id}", title_style="bold", border_style="light_slate_blue")
            table.add_column("Memory Type", style="cyan")
            table.add_column("Status", style="magenta")
            # ...existing code for checkpointer status...
            table.add_row("Persistent Memory (MongoDB)", "Available" if user_memory else "Unavailable")
            self.console.print(table)
        except Exception as e:
            logger.error(f"Error in log_memory_details: {e}")
            self.log_error(f"Error logging memory details: {e}")

    def log_prompt(self, prompt_data):
        table = Table(
            title="🧠 Complete AI Prompt Structure",
            title_style="bold bright_blue",
            box=box.ROUNDED,
            border_style="bright_blue",
            header_style="bold cyan"
        )
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Content", style="bright_white")
        if "system_prompt" in prompt_data:
            table.add_row("System Prompt", prompt_data["system_prompt"])
        if "kids_information" in prompt_data:
            table.add_row("Kids Information", prompt_data["kids_information"])
        if "ai_tone" in prompt_data:
            table.add_row("AI Tone", prompt_data["ai_tone"])
        if "conversation_context" in prompt_data:
            table.add_row("Conversation Context", prompt_data["conversation_context"])
        if "user_query" in prompt_data:
            table.add_row("User Query", prompt_data["user_query"])
        self.console.print(table)
