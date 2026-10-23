"""v-swing-lane-2026-10-23: watch-only daily swing lane (SPEC: review/swing_system/SPEC.md).

SHADOW ONLY. This package contains no order, broker, or position-mutation code.
It is imported only by core.swing_lane_gate, and only when ENABLE_SWING_SHADOW
is True. Disabled setups are never imported.
"""
