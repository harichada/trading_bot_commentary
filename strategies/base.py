from abc import ABC, abstractmethod
from typing import Optional

from core.models import TradingSignal


class TradingStrategyWithCommentary(ABC):
    """Base class for strategies with commentary"""

    def __init__(self, commentary_system):
        self.commentary = commentary_system

    @abstractmethod
    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        pass
