import json
import os
import logging
import time
import threading
import tempfile
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Set

import numpy as np
import pandas as pd

from core.models import CommentaryType
from core.config import Config

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

logger = logging.getLogger('TradingBot')

@dataclass
class TradingCommentary:
    """Represents a single commentary entry"""
    timestamp: datetime
    type: CommentaryType
    symbol: Optional[str]
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    importance: int = 5

    def to_dict(self):
        """Convert commentary to dictionary with proper type handling"""
        def make_serializable(obj):
            """Convert non-serializable objects to serializable format"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif hasattr(obj, '__dict__'):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        return {
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'symbol': self.symbol,
            'title': self.title,
            'message': self.message,
            'data': make_serializable(self.data) if self.data else None,
            'confidence': float(self.confidence) if self.confidence is not None else None,
            'importance': self.importance
        }

class CommentarySystem:
    """Manages trading commentary and explanations"""

    def __init__(self, max_history: int = 100):
        self.history: deque = deque(maxlen=max_history)
        self.subscribers: Set[callable] = set()
        self.console = console if RICH_AVAILABLE else None
        self._last_save_time = 0
        self._save_interval = 5  # Save at most once every 5 seconds
        self._save_lock = threading.Lock()

    def add_commentary(self, commentary: TradingCommentary):
        """Add new commentary and notify subscribers"""
        self.history.append(commentary)
        self._display_commentary(commentary)
        self._notify_subscribers(commentary)
        self._save_to_file(commentary)

    def _display_commentary(self, commentary: TradingCommentary):
        """Display commentary in terminal with rich formatting"""
        # Handle case where commentary might be a dict
        if isinstance(commentary, dict):
            # Convert dict back to TradingCommentary object
            try:
                commentary_type = CommentaryType(commentary.get('type', 'MARKET_ANALYSIS'))
            except ValueError:
                # If the type is not valid, default to MARKET_ANALYSIS
                commentary_type = CommentaryType.MARKET_ANALYSIS

            commentary = TradingCommentary(
                timestamp=datetime.fromisoformat(commentary['timestamp']) if isinstance(commentary.get('timestamp'), str) else commentary.get('timestamp', datetime.now()),
                type=commentary_type,
                symbol=commentary.get('symbol'),
                title=commentary.get('title', ''),
                message=commentary.get('message', ''),
                data=commentary.get('data', {}),
                confidence=commentary.get('confidence'),
                importance=commentary.get('importance', 5)
            )

        if not self.console:
            print(f"\n[{commentary.timestamp.strftime('%H:%M:%S')}] {commentary.title}")
            print(f"  {commentary.message}")
            return

        color_map = {
            CommentaryType.MARKET_ANALYSIS: "cyan",
            CommentaryType.SIGNAL_GENERATION: "yellow",
            CommentaryType.RISK_ASSESSMENT: "magenta",
            CommentaryType.DECISION: "green",
            CommentaryType.WARNING: "red",
            CommentaryType.OPPORTUNITY: "bright_green",
            CommentaryType.TECHNICAL: "blue",
            CommentaryType.PSYCHOLOGY: "purple",
            CommentaryType.ANOMALY: "bright_red",
            CommentaryType.INFO: "white",
            CommentaryType.ACCOUNT_UPDATE: "bright_cyan",
            CommentaryType.ERROR: "bright_red"
        }

        color = color_map.get(commentary.type, "white")

        title = f"[bold {color}]{commentary.title}[/bold {color}]"

        content = Text()
        content.append(commentary.message, style=f"{color}")

        if commentary.data:
            content.append("\n\n📊 Details:\n", style="bold")
            for key, value in commentary.data.items():
                if isinstance(value, (int, float)):
                    content.append(f"  • {key}: ", style="dim")
                    content.append(f"{value:.4f}" if isinstance(value, float) else str(value), style="bold")
                    content.append("\n")
                else:
                    content.append(f"  • {key}: {value}\n", style="dim")

        if commentary.confidence:
            conf_color = "green" if commentary.confidence > 0.7 else "yellow" if commentary.confidence > 0.4 else "red"
            content.append(f"\n💡 Confidence: ", style="dim")
            content.append(f"{commentary.confidence:.1%}", style=f"bold {conf_color}")

        panel = Panel(
            content,
            title=title,
            subtitle=f"[dim]{commentary.timestamp.strftime('%H:%M:%S')}[/dim]",
            border_style=color,
            expand=False
        )

        self.console.print(panel)

    def _notify_subscribers(self, commentary: TradingCommentary):
        """Notify all subscribers of new commentary"""
        for subscriber in self.subscribers:
            try:
                subscriber(commentary)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
                # Log more details for debugging
                import traceback
                logger.debug(f"Subscriber error traceback: {traceback.format_exc()}")

    def _save_to_file(self, commentary: TradingCommentary):
        """Save commentary to file for later analysis with robust error handling"""
        try:
            # Rate limiting - only save once every few seconds
            current_time = time.time()
            if current_time - self._last_save_time < self._save_interval:
                return  # Skip this save

            # Use lock to prevent concurrent saves
            if not self._save_lock.acquire(blocking=False):
                return  # Another save is in progress

            try:
                self._last_save_time = current_time

                # Ensure we have an absolute path
                commentary_path = Path(Config().COMMENTARY_LOG_PATH)
                if not commentary_path.is_absolute():
                    # Make it relative to the current working directory
                    commentary_path = Path.cwd() / commentary_path

                commentary_path = str(commentary_path)
                all_commentary = []

                # Try to load existing commentary with error recovery
                if Path(commentary_path).exists():
                    try:
                        with open(commentary_path, 'r') as f:
                            all_commentary = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Corrupted commentary file, attempting recovery: {e}")
                        # Try to recover valid JSON objects
                        try:
                            self._recover_commentary_file(commentary_path)
                            # Try loading again after recovery
                            with open(commentary_path, 'r') as f:
                                all_commentary = json.load(f)
                        except Exception as recovery_error:
                            logger.error(f"Recovery failed: {recovery_error}")
                            # Start fresh with backup
                            backup_path = Path(commentary_path).with_suffix('.json.corrupted')
                            Path(commentary_path).rename(backup_path)
                            logger.info(f"Backed up corrupted file to {backup_path}")
                            all_commentary = []

                # Ensure commentary can be serialized
                try:
                    commentary_dict = commentary.to_dict()
                    # Test serialization before adding
                    json.dumps(commentary_dict)
                    all_commentary.append(commentary_dict)
                except (TypeError, ValueError) as e:
                    logger.error(f"Commentary serialization error: {e}")
                    # Create a simplified version
                    commentary_dict = {
                        'timestamp': str(commentary.timestamp),
                        'type': str(commentary.type),
                        'symbol': str(commentary.symbol),
                        'title': str(commentary.title),
                        'message': str(commentary.message),
                        'importance': commentary.importance
                    }
                    all_commentary.append(commentary_dict)

                # Maintain size limit
                if len(all_commentary) > 1000:
                    all_commentary = all_commentary[-1000:]

                # Write with atomic operation
                commentary_path_obj = Path(commentary_path).resolve()

                try:
                    # Ensure directory exists
                    commentary_path_obj.parent.mkdir(parents=True, exist_ok=True)

                    # Create a unique temp file in the same directory
                    import tempfile
                    import os
                    fd, temp_path_str = tempfile.mkstemp(
                        suffix='.tmp',
                        prefix='commentary_',
                        dir=str(commentary_path_obj.parent)
                    )

                    try:
                        # Write to temp file using the file descriptor
                        with os.fdopen(fd, 'w') as f:
                            json.dump(all_commentary, f, indent=2, default=str)

                        # Atomic rename
                        os.replace(temp_path_str, str(commentary_path_obj))
                    except Exception as write_error:
                        # Close fd if still open
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                        raise write_error
                    finally:
                        # Clean up temp file if it exists
                        if os.path.exists(temp_path_str):
                            try:
                                os.unlink(temp_path_str)
                            except OSError:
                                pass

                except Exception as write_error:
                    logger.error(f"Error writing commentary file: {write_error}")
                    # Try direct write as fallback
                    try:
                        with open(commentary_path_obj, 'w') as f:
                            json.dump(all_commentary, f, indent=2, default=str)
                    except Exception as fallback_error:
                        logger.error(f"Fallback write also failed: {fallback_error}")
            finally:
                # Always release the lock
                self._save_lock.release()

        except Exception as e:
            logger.error(f"Error saving commentary: {e}")
            # Don't let save errors crash the system

    def _recover_commentary_file(self, filepath):
        """Attempt to recover corrupted commentary JSON file"""
        with open(filepath, 'r') as f:
            content = f.read()

        valid_objects = []
        lines = content.split('\n')
        current_obj = ""
        depth = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.strip() == '[':
                i += 1
                continue

            if line.strip() == '{':
                current_obj = line
                depth = 1
                i += 1

                while i < len(lines) and depth > 0:
                    line = lines[i]
                    current_obj += '\n' + line

                    # Simple depth tracking
                    depth += line.count('{') - line.count('}')
                    i += 1

                    if depth == 0:
                        try:
                            obj_str = current_obj.rstrip()
                            if obj_str.endswith(','):
                                obj_str = obj_str[:-1]
                            obj = json.loads(obj_str)
                            valid_objects.append(obj)
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
            else:
                i += 1

        if valid_objects:
            with open(filepath, 'w') as f:
                json.dump(valid_objects, f, indent=2)
            logger.info(f"Recovered {len(valid_objects)} commentary entries")

    def subscribe(self, callback: callable):
        """Subscribe to commentary updates"""
        self.subscribers.add(callback)
