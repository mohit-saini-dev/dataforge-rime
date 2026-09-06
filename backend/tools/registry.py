"""
Tool definition schemas, metadata, and function registry for LLM tool calling.
Compatible with standard OpenAI and LiveKit function definition specifications.
"""

import copy
from typing import Any, Callable, Coroutine, Dict, List, Optional
from backend.tools.mock_tools import (
    book_flight,
    cancel_booking,
    search_flights,
    search_hotels,
)


class ToolRegistryError(Exception):
    """Base exception for registry errors."""
    pass


class ToolNotFoundError(ToolRegistryError):
    """Raised when an unknown tool is invoked."""
    pass


class InvalidToolArgumentsError(ToolRegistryError):
    """Raised when arguments fail schema validation."""
    pass


# Function dispatch registry mapping tool names to coroutines
TOOL_HANDLERS: Dict[str, Callable[..., Coroutine[Any, Any, Dict[str, Any]]]] = {
    "search_flights": search_flights,
    "book_flight": book_flight,
    "cancel_booking": cancel_booking,
    "search_hotels": search_hotels,
}

# Runtime execution metadata for TurnController fences and timeout budgets
TOOL_METADATA: Dict[str, Dict[str, Any]] = {
    "search_flights": {
        "max_latency_ms": 1000,
        "idempotent": True,
        "mutates_state": False,
        "required_params": {"origin", "destination"},
    },
    "book_flight": {
        "max_latency_ms": 1500,
        "idempotent": False,
        "mutates_state": True,
        "required_params": {"flight_id", "passenger_name"},
    },
    "cancel_booking": {
        "max_latency_ms": 1200,
        "idempotent": True,
        "mutates_state": True,
        "required_params": {"booking_id"},
    },
    "search_hotels": {
        "max_latency_ms": 1000,
        "idempotent": True,
        "mutates_state": False,
        "required_params": {"city"},
    },
}

# Standard OpenAI / LiveKit function calling definitions
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between 3-letter IATA airport codes.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "origin": {
                        "type": "string",
                        "pattern": "^[A-Z]{3}$",
                        "description": "3-letter IATA departure airport code, e.g., DEL",
                    },
                    "destination": {
                        "type": "string",
                        "pattern": "^[A-Z]{3}$",
                        "description": "3-letter IATA arrival airport code, e.g., BOM",
                    },
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "Optional departure date in YYYY-MM-DD format",
                    },
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a ticket for a selected flight using its flight ID and passenger name.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "flight_id": {
                        "type": "string",
                        "pattern": "^FL-[0-9]{3}$",
                        "description": "The unique ID of the flight to book, e.g., FL-101",
                    },
                    "passenger_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 100,
                        "description": "Full name of the primary passenger",
                    },
                },
                "required": ["flight_id", "passenger_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": "Cancel an existing confirmed booking and initiate seat restoration.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "pattern": "^BK-[A-Z0-9]{6,8}$",
                        "description": "The unique booking reference ID, e.g., BK-A1B2C3",
                    },
                },
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search available hotel accommodations in a given destination city code.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "city": {
                        "type": "string",
                        "pattern": "^[A-Z]{3}$",
                        "description": "3-letter IATA city or airport code, e.g., BOM",
                    },
                    "nights": {
                        "type": "integer",
                        "description": "Number of nights for the stay (defaults to 1)",
                        "minimum": 1,
                        "maximum": 30,
                    },
                },
                "required": ["city"],
            },
        },
    },
]


def get_tool_handler(tool_name: str) -> Callable[..., Coroutine[Any, Any, Dict[str, Any]]]:
    """Retrieve the callable coroutine for a given tool name."""
    if tool_name not in TOOL_HANDLERS:
        raise ToolNotFoundError(f"Tool '{tool_name}' is not registered.")
    return TOOL_HANDLERS[tool_name]


def get_tool_metadata(tool_name: str) -> Dict[str, Any]:
    """Retrieve execution metadata (latency budget, mutation flag, required params)."""
    if tool_name not in TOOL_METADATA:
        raise ToolNotFoundError(f"Metadata for tool '{tool_name}' not found.")
    return TOOL_METADATA[tool_name]


def validate_tool_arguments(tool_name: str, args: Dict[str, Any]) -> None:
    """Validate that all required parameters are present before dispatch."""
    meta = get_tool_metadata(tool_name)
    required = meta.get("required_params", set())
    missing = required - set(args.keys())
    if missing:
        raise InvalidToolArgumentsError(
            f"Missing required parameters for '{tool_name}': {sorted(list(missing))}"
        )


def get_available_tool_definitions() -> List[Dict[str, Any]]:
    """Return a deep copy of schema definitions formatted for the LLM function call context."""
    return copy.deepcopy(TOOL_DEFINITIONS)