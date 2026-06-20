"""
api/models_router.py
--------------------
GET /models – returns the full catalog of supported LLM providers and their
available model IDs.

This endpoint is designed to be consumed by frontend clients to dynamically
populate a model-selector dropdown without hard-coding provider/model lists
in the UI.
"""

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.llm_service import MODELS_CATALOG
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    summary="List all supported LLM providers and models",
    response_description=(
        "A dictionary keyed by provider ID, each containing a list of available "
        "model objects with id, name, description, and default flag."
    ),
)
async def list_models() -> JSONResponse:
    """
    Return the complete catalog of LLM providers and their supported models.

    Response shape
    --------------
    ```json
    {
      "providers": {
        "anthropic": {
          "display_name": "Anthropic (Claude)",
          "requires_key": true,
          "models": [
            { "id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku",
              "description": "Fast & efficient", "default": true },
            ...
          ]
        },
        "nvidia": {
          "display_name": "NVIDIA (Free Models)",
          "requires_key": false,
          "models": [ ... ]
        },
        ...
      }
    }
    ```

    The ``requires_key`` flag indicates whether a user-supplied API key is
    mandatory (``true``) or optional / not needed (``false``).
    """
    log.debug("GET /models – returning provider catalog")

    provider_meta: dict[str, Any] = {
        "anthropic": {
            "display_name": "Anthropic (Claude)",
            "requires_key": True,
            "key_hint": "Get your key at https://console.anthropic.com",
        },
        "openai": {
            "display_name": "OpenAI (ChatGPT)",
            "requires_key": True,
            "key_hint": "Get your key at https://platform.openai.com",
        },
        "gemini": {
            "display_name": "Google (Gemini)",
            "requires_key": True,
            "key_hint": "Get your key at https://aistudio.google.com",
        },
        "nvidia": {
            "display_name": "NVIDIA (Free Models)",
            "requires_key": False,
            "key_hint": "Optional: get a free key at https://build.nvidia.com for higher throughput",
        },
        "mock": {
            "display_name": "Mock (Offline Testing)",
            "requires_key": False,
            "key_hint": None,
        },
    }

    response: dict[str, Any] = {"providers": {}}
    for provider_id, models in MODELS_CATALOG.items():
        meta = provider_meta.get(provider_id, {})
        response["providers"][provider_id] = {
            **meta,
            "models": models,
        }

    return JSONResponse(content=response)
