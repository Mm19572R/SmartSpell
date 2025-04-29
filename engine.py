import re
from spellchecker import SpellChecker
from analyze_grammar import analyze_grammar

spell = SpellChecker()

def run_spelling(text: str) -> list[dict]:
    suggestions = []
    words = re.findall(r'\w+', text)
    for w in words:
        corr = spell.correction(w)
        if corr and corr.lower() != w.lower():
            suggestions.append({"type": "spelling", "original": w, "suggested": corr})
    return suggestions

def run_grammar(text: str) -> list[dict]:
    return [{"type": "grammar", "message": msg} for msg in analyze_grammar(text)["issues"]]

def correct_text(text: str):
    suggestions = run_spelling(text) + run_grammar(text)
    corrected = text
    for s in run_spelling(text):
        pattern = rf'\b{s["original"]}\b'
        corrected = re.sub(pattern, s["suggested"], corrected, 1)
    return corrected, suggestions
