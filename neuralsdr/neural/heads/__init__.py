"""neural/heads/__init__.py"""
from .analog_head  import AnalogHead
from .digital_head import DigitalHead
from .squelch_head import SquelchHead

__all__ = ["AnalogHead", "DigitalHead", "SquelchHead"]
