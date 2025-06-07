# Devin/modules/ai_connector.py
# Purpose: Defines a generic, abstract interface (contract) for connecting
#          to any AI model, ensuring all AI modules are interchangeable.
# Generic AI connection 🔌

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Configure basic logging
logger = logging.getLogger("AIConnector")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

@dataclass
class AIRequest:
    """A standardized representation of a request to an AI model."""
    messages: List[Dict[str, str]]
    model: str
    config: Dict[str, Any] = field(default_factory=dict) # e.g., temperature, max_tokens

@dataclass
class AIResponse:
    """A standardized representation of a response from an AI model."""
    content: Optional[str]
    is_success: bool
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict] = None # To store the original provider response

class AIGenericConnector(ABC):
    """
    An Abstract Base Class (ABC) that defines the "contract" for any AI connector.
    Any class that interacts with an AI model (like ChatGPT, Gemini, Claude) should
    inherit from this class and implement its abstract methods.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Connector for model '{self.model_name}' of type '{type(self).__name__}' is being initialized.")
        super().__init__()

    @abstractmethod
    def get_chat_completion(self, request: AIRequest) -> AIResponse:
        """
        Sends a request to the AI model and gets a chat completion.

        Args:
            request (AIRequest): The standardized request object.

        Returns:
            AIResponse: The standardized response object.
        """
        pass

    @abstractmethod
    def count_tokens_conceptual(self, text_or_messages: Any) -> int:
        """
        Conceptually counts the number of tokens for a given text or message list.

        Args:
            text_or_messages: The content to be tokenized.

        Returns:
            int: The conceptual number of tokens.
        """
        pass

    def get_provider_name(self) -> str:
        """Returns the name of the AI provider for this connector."""
        # This can be overridden but provides a default based on the class name.
        return self.__class__.__name__.replace("Connector", "")

# --- Example Concrete Implementation ---
# This demonstrates how a real module like ChatGPTModule would implement the interface.

class SimpleEchoConnector(AIGenericConnector):
    """
    A simple, concrete implementation of the AIGenericConnector for testing.
    It doesn't connect to any real AI; it just echoes the user's message back.
    """
    def __init__(self, model_name: str = "echo-v1"):
        super().__init__(model_name)

    def get_chat_completion(self, request: AIRequest) -> AIResponse:
        """Implements the abstract method for the Echo connector."""
        logger.info(f"SimpleEchoConnector processing request for model '{request.model}'...")
        if not request.messages or request.messages[-1]["role"] != "user":
            error_msg = "Invalid request: Last message must be from 'user'."
            logger.error(error_msg)
            return AIResponse(content=None, is_success=False, error_message=error_msg)

        last_user_message = request.messages[-1]["content"]
        echo_content = f"Echo from {self.model_name}: '{last_user_message}'"
        
        return AIResponse(
            content=echo_content,
            is_success=True,
            finish_reason="stop",
            raw_response={"simulated": True, "echoed_text": last_user_message}
        )
    
    def count_tokens_conceptual(self, text_or_messages: Any) -> int:
        """Implements the abstract method for the Echo connector."""
        # A simple token counting simulation (e.g., words)
        if isinstance(text_or_messages, str):
            return len(text_or_messages.split())
        elif isinstance(text_or_messages, list):
            total_words = 0
            for msg in text_or_messages:
                total_words += len(msg.get("content", "").split())
            return total_words
        return 0


# --- Example Usage ---
if __name__ == "__main__":
    print("=========================================================")
    print("=== Generic AI Connector Prototype 🔌 ===")
    print("=========================================================")

    # 1. Demonstrate that the Abstract Base Class cannot be instantiated directly
    print("\n--- Attempting to instantiate the abstract class (should fail) ---")
    try:
        generic_connector = AIGenericConnector(model_name="abstract-model")
    except TypeError as e:
        print(f"  SUCCESS: Caught expected TypeError: {e}")

    # 2. Instantiate and use the concrete implementation
    print("\n\n--- Using the concrete 'SimpleEchoConnector' implementation ---")
    echo_connector = SimpleEchoConnector()
    
    # Create a standardized request
    my_request = AIRequest(
        messages=[
            {"role": "system", "content": "You are an echo bot."},
            {"role": "user", "content": "Hello, world! Can you hear me?"}
        ],
        model="echo-v1",
        config={"temperature": 0.1} # Config is passed but not used by echo bot
    )
    
    # Get a response
    response = echo_connector.get_chat_completion(my_request)
    
    print(f"\n  Provider: {echo_connector.get_provider_name()}")
    print(f"  Request Model: {my_request.model}")
    print(f"  Response Success: {response.is_success}")
    print(f"  Response Content: {response.content}")
    print(f"  Finish Reason: {response.finish_reason}")

    # 3. Demonstrate token counting
    print("\n\n--- Demonstrating conceptual token counting ---")
    messages_for_counting = my_request.messages
    token_count = echo_connector.count_tokens_conceptual(messages_for_counting)
    print(f"  Conceptual token count for messages: {token_count}")
    
    string_for_counting = "This is another test string."
    token_count_str = echo_connector.count_tokens_conceptual(string_for_counting)
    print(f"  Conceptual token count for a string: {token_count_str}")

    print("\n=========================================================")
    print("=== AI Connector Prototype Complete ===")
    print("=========================================================")
