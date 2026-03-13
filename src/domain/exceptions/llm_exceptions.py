class LLMError(Exception):
    """Base exception for LLM related errors."""

    pass


class RateLimitError(LLMError):
    """Raised when the provider rate limit is exceeded."""

    pass


class CreditExhaustionError(LLMError):
    """Raised when the provider credit is exhausted."""

    pass


class FormatValidationError(LLMError):
    """Raised when the provider response doesn't match the requested schema."""

    pass


class ProviderUnavailableError(LLMError):
    """Raised when the provider is down or unreachable."""

    pass


class ChainExhaustedError(LLMError):
    """Raised when all providers in the chain have failed."""

    pass
