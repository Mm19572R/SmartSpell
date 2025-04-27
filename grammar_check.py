import spacy
from spellchecker import SpellChecker

nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

def analyze_grammar(text):
    doc = nlp(text)
    issues = []

    # Check grammar-related stuff (sentence structure, etc.)

    # Check spelling for each word
    for token in doc:
        if not token.is_punct and not token.is_space:
            word_lower = token.text.lower()
            if word_lower not in spell.word_frequency:
                issues.append(f"Possible spelling mistake in: '{token.text}'")
    
    return {"issues": issues}
