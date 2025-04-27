import spacy
from spellchecker import SpellChecker

# Load small English model for POS, dependencies, etc.
nlp = spacy.load("en_core_web_sm")
# Fallback spelling checker
token_spell = SpellChecker()

def analyze_grammar(text: str) -> dict:
    """
    Analyze text for basic grammar and spelling issues using spaCy and SpellChecker.

    Returns:
        {"issues": [list of issue messages]}
    """
    doc = nlp(text)
    issues = []

    # 1) Spelling mistakes
    for token in doc:
        if not token.is_punct and not token.is_space:
            word = token.text
            # ignore proper nouns (names)
            if not (word[0].isupper() and token.pos_ == 'PROPN'):
                if word.lower() not in token_spell.word_frequency:
                    issues.append(f"Possible spelling mistake in: '{word}'")

    # 2) Missing subject: root verb without a nominal subject
    for token in doc:
        if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
            has_subj = any(child.dep_ in ('nsubj', 'nsubjpass') for child in token.children)
            if not has_subj:
                issues.append("Sentence may be missing a subject.")

    # 3) Passive voice detection
    for token in doc:
        if token.dep_ == 'auxpass':
            issues.append("Possible passive voice detected. Consider using active voice.")
            break

    # 4) Repeated word detection
    for prev, curr in zip(doc, doc[1:]):
        if prev.text.lower() == curr.text.lower():
            issues.append(f"Repeated word: '{prev.text}'.")
            break

    # 5) Punctuation / ending
    if text and text[-1] not in '.!?':
        issues.append("Sentence should end with proper punctuation (., !, or ?).")

    return {"issues": issues}
