import re
import spacy
from spellchecker import SpellChecker
from lemminflect import getInflection
import language_tool_python
from happytransformer import HappyTextToText
from difflib import SequenceMatcher

# Initialize tools
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

# Load real contractions
spell.word_frequency.load_words([
    "aren't", "isn't", "don't", "doesn't", "won't", "shouldn't", "couldn't", "wouldn't",
    "didn't", "hasn't", "haven't", "hadn't", "wasn't", "weren't", "it's", "I'm", "they're",
    "we're", "you're", "I've", "you've", "they've", "she's", "he's", "who's", "that's", "what's",
    "there's", "let's", "where's", "how's", "y'all", "ain't"
])

tool = language_tool_python.LanguageTool('en', remote_server='http://localhost:8081')
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

def correct_spelling(text):
    text = text.replace("’", "'").replace("‘", "'")
    words = re.findall(r'\b\w+\b', text)
    corrections = []

    manual_contractions = {
        "dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't",
        "wasnt": "wasn't", "werent": "weren't", "hasnt": "hasn't", "hadnt": "hadn't",
        "wont": "won't", "wouldnt": "wouldn't", "couldnt": "couldn't", "shouldnt": "shouldn't",
        "cant": "can't", "mustnt": "mustn't", "didnt": "didn't"
    }

    for word in words:
        lower = word.lower()
        if lower in manual_contractions:
            corrections.append({'type': 'spelling', 'original': word, 'suggested': manual_contractions[lower]})
        elif lower not in spell:
            suggestion = spell.correction(word)
            if suggestion and suggestion.lower() != lower:
                corrections.append({'type': 'spelling', 'original': word, 'suggested': suggestion})

    return corrections

def correct_morphology(text):
    doc = nlp(text)
    corrections = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            verb = token.head
            subj_num = token.morph.get("Number")
            if not subj_num:
                continue
            suggested = None
            if subj_num == ["Sing"] and token.text.lower() in ["he", "she", "it"] and verb.tag_ != "VBZ":
                suggested = getInflection(verb.lemma_, "VBZ")
            elif subj_num == ["Plur"] and verb.tag_ == "VBZ":
                suggested = getInflection(verb.lemma_, "VBP")
            if suggested and suggested[0] != verb.text:
                corrections.append({'type': 'morphology', 'original': verb.text, 'suggested': suggested[0]})
    return corrections

def correct_grammar(text):
    try:
        matches = tool.check(text)
    except Exception as e:
        return [{'type': 'error', 'original': '', 'suggested': '', 'message': f'Grammar check failed: {e}'}]

    corrections = []
    sentences = list(nlp(text).sents)

    for match in matches:
        offset = match.offset
        length = match.errorLength
        replacement = match.replacements[0] if match.replacements else None
        if not replacement:
            continue

        original = text[offset:offset + length].strip()
        if not original or original == replacement:
            continue

        sentence_text = ""
        for sent in sentences:
            if sent.start_char <= offset < sent.end_char:
                sentence_text = sent.text.strip()
                break
        else:
            continue

        corrected_sentence = sentence_text.replace(original, replacement, 1).strip()
        if corrected_sentence and corrected_sentence != sentence_text:
            corrections.append({
                'type': 'grammar',
                'original': sentence_text,
                'suggested': corrected_sentence,
                'message': match.message
            })

    return corrections

def correct_gpt(text):
    sentences = list(nlp(text).sents)
    suggestions = []
    for sent in sentences:
        raw = sent.text.strip()
        result = happy_tt.generate_text("grammar: " + raw)
        corrected = result.text.strip()

        if (
            not corrected or
            corrected.lower().strip(". ") == raw.lower().strip(". ") or
            corrected == raw
        ):
            continue

        suggestions.append({
            'type': 'grammar',
            'original': raw,
            'suggested': corrected,
            'message': "T5 tone/style enhancement"
        })

    return suggestions

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def correct_text(text):
    spelling = correct_spelling(text)
    morphology = correct_morphology(text)
    grammar = correct_grammar(text)
    gpt = correct_gpt(text)

    all_grammar = grammar + gpt

    # Group grammar corrections by original sentence
    grammar_by_sentence = {}
    for c in all_grammar:
        key = c["original"].strip()
        if key not in grammar_by_sentence:
            grammar_by_sentence[key] = c
        else:
            # Prefer the suggestion that changes more (longer difference)
            prev = grammar_by_sentence[key]
            prev_diff = len(prev["original"]) - len(prev["suggested"])
            new_diff = len(c["original"]) - len(c["suggested"])
            if abs(new_diff) > abs(prev_diff):
                grammar_by_sentence[key] = c

    # Combine all corrections
    corrections = spelling + morphology + list(grammar_by_sentence.values())

    # Apply corrections
    corrected_text = text
    for c in corrections:
        o = c.get("original", "").strip()
        s = c.get("suggested", "").strip()
        if o and s and o != s:
            corrected_text = corrected_text.replace(o, s, 1)

    return corrected_text, corrections
