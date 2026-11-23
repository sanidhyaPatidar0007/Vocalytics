"""Application exception hierarchy.

Every error carries a ``user_message`` that is safe to show in the UI: it
never contains file system paths, secrets, SQL or stack traces. Internal
detail goes to the server log only.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for expected, user-presentable failures."""

    default_message = "The operation could not be completed."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None) -> None:
        self.user_message = message or self.default_message
        self.details: dict[str, Any] = details or {}
        super().__init__(self.user_message)


class ConfigurationError(AppError):
    default_message = "The application is not configured correctly."


class ValidationError(AppError):
    default_message = "The input is not valid."


class FileValidationError(ValidationError):
    default_message = "The file could not be accepted."


class AuthenticationError(AppError):
    default_message = "Invalid username or password."


class AccountLockedError(AuthenticationError):
    default_message = "Too many failed sign-in attempts. Try again later."


class SessionExpiredError(AuthenticationError):
    default_message = "Your session has expired. Please sign in again."


class PermissionDeniedError(AppError):
    default_message = "You do not have permission to perform this action."


class FeatureNotAvailableError(PermissionDeniedError):
    default_message = "This feature is not included in your organization's plan."


class NotFoundError(AppError):
    """Raised for missing records AND for records owned by another tenant.

    Using one error for both cases avoids leaking whether a resource exists.
    """

    default_message = "The requested item was not found."


class ConflictError(AppError):
    default_message = "The request conflicts with the current state."


class PlanLimitExceededError(AppError):
    default_message = "This action would exceed your plan's limits."


class StorageQuotaExceededError(PlanLimitExceededError):
    default_message = "Your organization's storage quota would be exceeded."


class ResourceLimitError(AppError):
    default_message = "Not enough memory or disk space is available for this operation."


class OperationCancelledError(AppError):
    default_message = "The operation was cancelled. No changes were saved."


class OperationTimeoutError(AppError):
    default_message = "The operation took too long and was stopped. No changes were saved."


class ExternalServiceError(AppError):
    default_message = "An external service is unavailable."


class LLMUnavailableError(ExternalServiceError):
    default_message = "The local AI model is unavailable."


class LLMResponseError(ExternalServiceError):
    default_message = "The local AI model returned a response that could not be used."


class AIBusyError(ExternalServiceError):
    default_message = "The AI engine is busy with another request. Please try again shortly."


class PlanValidationError(ValidationError):
    default_message = "The analysis plan is not valid."


class VoiceError(AppError):
    default_message = "Voice input could not be processed."


class MicrophoneError(VoiceError):
    default_message = "The microphone could not be accessed. Check the microphone permission."


class EmptyAudioError(VoiceError):
    default_message = "No speech was detected in the recording."


class TranscriptionTimeoutError(VoiceError):
    default_message = "Speech recognition took too long. Please try a shorter recording."


class UnsupportedLanguageError(VoiceError):
    default_message = "The selected language is not supported."


class SpeechEngineUnavailableError(VoiceError):
    default_message = "No speech recognition engine is available."


class EmailDeliveryError(ExternalServiceError):
    default_message = "The e-mail could not be sent."
