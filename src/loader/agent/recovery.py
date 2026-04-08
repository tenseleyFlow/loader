"""Legacy compatibility exports for runtime-owned recovery services."""

from ..runtime.recovery import (
    ErrorCategory,
    RECOVERY_PROMPT,
    RecoveryContext,
    ToolAttempt,
    categorize_error,
    format_failure_message,
    format_recovery_prompt,
    get_recovery_hints,
)
