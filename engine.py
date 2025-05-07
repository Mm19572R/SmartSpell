import re
from spellchecker import SpellChecker
import language_tool_python
from analyze_grammar import analyze_grammar
import logging
import os
import shutil
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize spell checker with custom dictionary
spell = SpellChecker()

def add_custom_dictionary():
    """Add custom words and corrections to the spell checker."""
    # Add valid words to the dictionary
    spell.word_frequency.load_words([
        'groceries', 'grocery', "grocer's",
        'bought', 'went', 'gone',
        'windowsill', 'noise', 'homework'
    ])
    
    # Remove common misspellings from the dictionary
    spell.word_frequency.remove_words([
        'grocerys', 'buyed', 'goed',
        'windowsil', 'noize', 'homwork', 
        'nite'
    ])

def initialize_language_tool(max_retries=3):
    """Initialize LanguageTool with retries and proper error handling."""
    for attempt in range(max_retries):
        try:
            return language_tool_python.LanguageToolPublicAPI('en-US')
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise RuntimeError("Failed to initialize LanguageTool after multiple attempts") from e

try:
    tool = initialize_language_tool()
except Exception as e:
    logging.error("Failed to initialize LanguageTool", exc_info=True)
    tool = None  # Allow the program to continue without LanguageTool

def run_spelling(text: str) -> list[dict]:
    """Identify spelling errors in the given text."""
    # Split text into words while preserving case
    words = re.findall(r'\b\w+\b', text)
    suggestions = []
    
    for word in words:
        # Skip proper nouns (detected by capitalization in middle of sentence)
        is_proper = not word.islower() and not word.isupper() and word != words[0]
        
        # Handle possessive forms
        base_word = word.lower().rstrip("'s")
        
        # Check if the word (lowercase) is misspelled
        if not is_proper and base_word not in spell and not word.isnumeric():
            correction = spell.correction(base_word)
            candidates = spell.candidates(base_word) or []  # Ensure candidates is not None
            candidates = list(candidates)  # Convert to list if not empty
            
            # Handle plural forms
            if word.lower().endswith('s'):
                singular = word[:-1]
                if singular in spell:
                    correction = singular + 's'
                    candidates.append(correction)
            
            # Add possessive forms to candidates if applicable
            if word.endswith("'s"):
                possessive_candidates = [c + "'s" for c in candidates]
                candidates.extend(possessive_candidates)
            
            if correction and correction.lower() != word.lower():
                # Remove duplicates and limit to top 3 suggestions
                unique_candidates = list(dict.fromkeys(candidates))[:3]
                suggestions.append({
                    "type": "spelling",
                    "original": word,
                    "suggested": unique_candidates if unique_candidates else [correction]
                })
    
    return suggestions

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text to normalize capitalization, punctuation, and spacing.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Capitalize the first letter of each sentence and ensure proper spacing
    sentences = re.split(r'(\.|\?|!)', text)
    sentences = [s.strip().capitalize() + p for s, p in zip(sentences[::2], sentences[1::2]) if s]
    preprocessed_text = ' '.join(sentences)

    # Ensure proper spacing around punctuation
    preprocessed_text = re.sub(r'\s+([?.!,])', r'\1', preprocessed_text)
    preprocessed_text = re.sub(r'([?.!,])(?=[^\s])', r'\1 ', preprocessed_text)

    return preprocessed_text

def detect_common_issues(text: str) -> list[dict]:
    """
    Detect common grammar issues like subject-verb agreement and tense errors manually.

    Args:
        text (str): The input text to analyze.

    Returns:
        list[dict]: A list of detected issues with suggestions.
    """
    issues = []
    words = text.split()

    # Example: Detect "I buying" and suggest "I am buying"
    for i, word in enumerate(words[:-1]):
        if word.lower() == "i" and words[i + 1].endswith("ing"):
            issues.append({
                "type": "grammar",
                "message": "Missing auxiliary verb (e.g., 'am').",
                "original": f"{word} {words[i + 1]}",
                "suggested": [f"I am {words[i + 1]}"]
            })

    return issues

def run_grammar(text: str) -> list[dict]:
    """
    Identify grammar issues in the given text, including tense and agreement errors, and provide suggestions.

    Args:
        text (str): The input text to check for grammar issues.

    Returns:
        list[dict]: A list of suggestions for grammar corrections.
    """
    # Preprocess the text before grammar check
    text = preprocess_text(text)
    logging.debug(f"Preprocessed text for grammar check: {text}")

    suggestions = []

    # Use LanguageTool for grammar analysis
    try:
        matches = tool.check(text)
        logging.debug(f"Grammar matches: {len(matches)} found")

        for match in matches:
            suggestion = {
                "type": "grammar",
                "message": match.message,
                "original": text[match.offset : match.offset + match.errorLength],
                "suggested": match.replacements if match.replacements else ["No suggestion"]
            }

            # Highlight tense and agreement issues
            if "tense" in match.message.lower() or "agreement" in match.message.lower():
                suggestion["highlight"] = "Tense/Agreement Issue"

            logging.debug(f"Grammar suggestion: {suggestion}")
            suggestions.append(suggestion)

    except Exception as e:
        logging.error("Failed to analyze grammar using LanguageTool", exc_info=True)
        raise RuntimeError("Failed to analyze grammar using LanguageTool") from e

    # Add fallback manual checks for common issues
    manual_issues = detect_common_issues(text)
    suggestions.extend(manual_issues)

    # Log final suggestions for debugging
    logging.debug(f"Final grammar suggestions: {suggestions}")

    # Fallback: Add a warning if no grammar issues are detected
    if not suggestions:
        logging.warning("No grammar issues detected. Consider reviewing the input text manually.")

    return suggestions

def process_grammar_suggestion(issue: str) -> dict:
    """Process a grammar issue message into a structured suggestion."""
    suggestion = {
        "type": "grammar",
        "message": issue,
        "original": None,
        "suggested": []
    }
    
    # Extract original and suggested text from messages like "Use X instead of Y"
    if "Use '" in issue and "' instead of '" in issue:
        parts = issue.split("' instead of '")
        if len(parts) == 2:
            suggested = parts[0].split("'")[1] if "'" in parts[0] else None
            original = parts[1].split("'")[0] if "'" in parts[1] else None
            if suggested and original:
                suggestion.update({
                    "original": original.strip(),
                    "suggested": [suggested.strip()]
                })
    elif "instead of" in issue:
        parts = issue.split("instead of")
        if len(parts) == 2:
            suggested = parts[0].split("'")[1] if "'" in parts[0] else None
            original = parts[1].split("'")[1] if "'" in parts[1] else None
            if suggested and original:
                suggestion.update({
                    "original": original.strip(),
                    "suggested": [suggested.strip()]
                })
    
    return suggestion

def correct_text(text: str) -> tuple[str, list[dict]]:
    """Analyze and correct text for both spelling and grammar errors."""
    # Initialize custom dictionary
    add_custom_dictionary()
    
    all_suggestions = []
    corrected = text
    
    # Run spelling check first
    spelling_suggestions = run_spelling(text)
    logging.debug(f"Spelling suggestions: {spelling_suggestions}")
    all_suggestions.extend(spelling_suggestions)
    
    # Run grammar checks
    try:
        # Get grammar issues from our custom analyzer
        grammar_issues = analyze_grammar(text)
        logging.debug(f"Grammar issues: {grammar_issues}")
        
        for issue in grammar_issues.get("issues", []):
            suggestion = process_grammar_suggestion(issue)
            if suggestion["original"] or suggestion["message"]:
                all_suggestions.append(suggestion)
        
    except Exception as e:
        logging.error(f"Grammar check failed: {str(e)}", exc_info=True)
    
    # Validate suggestions to ensure they are contextually appropriate
    validated_suggestions = []
    for s in all_suggestions:
        if s.get("original") and s.get("suggested"):
            if isinstance(s["suggested"], list) and s["suggested"]:
                validated_suggestions.append(s)

    # Apply corrections more robustly, ensuring article corrections are handled
    for s in validated_suggestions:
        if s.get("original") and s.get("suggested"):
            corrected = re.sub(rf"\b{s['original']}\b", s['suggested'][0], corrected)
    
    logging.debug(f"Found {len(all_suggestions)} total suggestions")
    logging.debug(f"Original text: {text}")
    logging.debug(f"Corrected text: {corrected}")
    
    return corrected, all_suggestions