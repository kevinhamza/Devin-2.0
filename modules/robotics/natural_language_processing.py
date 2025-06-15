# Devin/modules/robotics/natural_language_processing.py
# Purpose: Provides Natural Language Processing (NLP) to understand user
#          commands by extracting intent and entities from raw text.

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

try:
    import spacy
    from spacy.tokens.doc import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy, Doc = None, None

# Configure basic logging
logger = logging.getLogger("NLPProcessor")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

class Intent(Enum):
    """Enumeration of recognized user intents."""
    FIND_OBJECT = auto()
    PICK_UP_OBJECT = auto()
    MOVE_ROBOT = auto()
    QUERY_STATUS = auto()
    GREETING = auto()
    UNKNOWN = auto()

@dataclass
class StructuredCommand:
    """A structured representation of a user's command."""
    intent: Intent
    entities: Dict[str, Any] = field(default_factory=dict)
    original_text: str

class NLPProcessor:
    """
    Processes natural language text to extract structured commands.
    """
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initializes the NLP processor by loading a spaCy model.
        """
        if not SPACY_AVAILABLE:
            self.nlp = None
            logger.error("spaCy library not found! Please run: 'pip install spacy'.")
            logger.error("Then download the model: 'python -m spacy download en_core_web_sm'")
            return
            
        try:
            logger.info(f"Loading spaCy model '{spacy_model}'...")
            self.nlp = spacy.load(spacy_model)
            logger.info("NLP Processor initialized successfully.")
        except OSError:
            logger.error(f"spaCy model '{spacy_model}' not found.")
            logger.error(f"Please run: python -m spacy download {spacy_model}")
            self.nlp = None
            
        # Define keywords for intent mapping
        self.intent_keywords = {
            Intent.FIND_OBJECT: ["find", "locate", "where is", "see"],
            Intent.PICK_UP_OBJECT: ["pick up", "get", "grab", "take", "retrieve"],
            Intent.MOVE_ROBOT: ["move to", "go to", "navigate"],
            Intent.QUERY_STATUS: ["status", "how are you", "report"],
            Intent.GREETING: ["hello", "hi", "hey"],
        }
        
    def _determine_intent(self, doc: Doc) -> Intent:
        """Determines the primary intent from the processed text."""
        text = doc.text.lower()
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return intent
        return Intent.UNKNOWN
        
    def _extract_entities(self, doc: Doc) -> Dict[str, Any]:
        """Extracts entities using spaCy's NER and custom rule-based logic."""
        entities = {}
        
        # 1. Use spaCy's built-in Named Entity Recognition (NER)
        for ent in doc.ents:
            entities[ent.label_.lower()] = ent.text
            
        # 2. Custom rule-based entity extraction
        # This is where domain-specific knowledge is added.
        colors = ["red", "blue", "green", "yellow", "black", "white"]
        
        for token in doc:
            # Extract color attributes
            if token.lemma_ in colors:
                # Find the noun the color is describing
                head_noun = token.head
                if head_noun.pos_ == "NOUN":
                     # Check if we already have an object, if not, add it
                    if "object" not in entities:
                        entities["object"] = head_noun.text
                    # Add the color attribute
                    entities["color"] = token.lemma_
                    
            # A simple way to find the main object of a "pick up" command
            if token.lemma_ in self.intent_keywords[Intent.PICK_UP_OBJECT]:
                # Find the direct object (dobj) of the verb
                for child in token.children:
                    if child.dep_ == "dobj":
                        entities["object"] = child.text
                        break # Take the first direct object
                        
        return entities

    def process_command(self, text: str) -> Optional[StructuredCommand]:
        """
        Processes a raw text command into a structured format.

        Returns:
            Optional[StructuredCommand]: The structured command, or None if processing fails.
        """
        if not self.nlp:
            logger.error("Cannot process command: NLP model not loaded.")
            return None
            
        logger.info(f"Processing text: '{text}'")
        doc = self.nlp(text)
        
        intent = self._determine_intent(doc)
        entities = self._extract_entities(doc)
        
        structured_command = StructuredCommand(
            intent=intent,
            entities=entities,
            original_text=text
        )
        
        logger.info(f"Processed command: Intent={intent.name}, Entities={entities}")
        return structured_command

# --- Example Usage ---
if __name__ == "__main__":
    print("=========================================================")
    print("=== Natural Language Processing (NLP) Prototype 🧠💬 ===")
    print("=========================================================")
    
    if not SPACY_AVAILABLE:
        print("\n'spaCy' library not found. Please install it and download a model to run this demo.")
    else:
        nlp_processor = NLPProcessor()
        
        if nlp_processor.nlp:
            commands_to_test = [
                "Devin, could you please pick up the red bottle for me?",
                "Where is the nearest charging station?",
                "Hey Devin, what's your status?",
                "Find a person near the desk.",
                "Move to the kitchen.",
                "Grab the wrench."
            ]
            
            print("\n--- Testing a series of commands ---")
            for command in commands_to_test:
                result = nlp_processor.process_command(command)
                if result:
                    print(f"\nOriginal: '{result.original_text}'")
                    print(f"  -> Intent: {result.intent.name}")
                    print(f"  -> Entities:")
                    if result.entities:
                        for key, value in result.entities.items():
                            print(f"     - {key.capitalize()}: {value}")
                    else:
                        print("     - None")
                    print("-" * 20)

    print("\n=========================================================")
    print("=== NLP Prototype Complete ===")
    print("=========================================================")
