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
    grammar_issues = analyze_grammar(text)["issues"]
    suggestions = []
    
    for msg in grammar_issues:
        suggestion = {"type": "grammar", "message": msg}
        
        # Extract correction information for subject-verb agreement issues
        if "Subject-verb agreement:" in msg:
            match = re.search(r"'(.+?)' instead of '(.+?)'", msg)
            if match:
                correct_phrase, incorrect_phrase = match.groups()
                suggestion["original"] = incorrect_phrase
                suggestion["suggested"] = correct_phrase
        
        suggestions.append(suggestion)
    
    return suggestions

def correct_text(text: str):
    suggestions = run_spelling(text) + run_grammar(text)
    corrected = text
    
    # Process grammar corrections first (they might contain multiple words)
    for s in [s for s in suggestions if s["type"] == "grammar"]:
        if "original" in s and "suggested" in s:
            # Use a more precise replacement for exact matches
            corrected = corrected.replace(s["original"], s["suggested"])
    
    # Then process spelling corrections
    for s in [s for s in suggestions if s["type"] == "spelling"]:
        pattern = rf'\b{re.escape(s["original"])}\b'
        corrected = re.sub(pattern, s["suggested"], corrected, 1)
    
    return corrected, suggestions