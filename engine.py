import re
import spacy
from spellchecker import SpellChecker
from lemminflect import getInflection
import language_tool_python
from happytransformer import HappyTextToText
import traceback

nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()
tool = language_tool_python.LanguageTool('en', remote_server='http://localhost:8081')
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

def correct_spelling(text):
    words = re.findall(r'\b\w+\b', text)
    corrections = []
    for word in words:
        if word.lower() not in spell:
            suggestion = spell.correction(word)
            if suggestion and suggestion.lower() != word.lower():
                corrections.append({
                    'type': 'spelling',
                    'original': word,
                    'suggested': suggestion
                })
    return corrections

def correct_grammar(text):
    try:
        matches = tool.check(text)
    except Exception as e:
        return [{
            'type': 'error',
            'original': '',
            'suggested': '',
            'message': f'Grammar check failed: {e}'
        }]

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

        # find which sentence the error belongs to
        sentence_text = ""
        for sent in sentences:
            if sent.start_char <= offset < sent.end_char:
                sentence_text = sent.text.strip()
                break

        corrected_sentence = sentence_text.replace(original, replacement, 1).strip()

        if corrected_sentence and corrected_sentence != sentence_text:
            corrections.append({
                'type': 'grammar',
                'original': sentence_text,
                'suggested': corrected_sentence,
                'message': match.message
            })

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
                corrections.append({
                    'type': 'morphology',
                    'original': verb.text,
                    'suggested': suggested[0]
                })
    return corrections

def correct_gpt(text):
    result = happy_tt.generate_text("grammar: " + text)
    corrected = result.text.strip()

    if (
        not corrected or
        corrected.lower().strip(". ") == text.lower().strip(". ") or
        corrected == text
    ):
        return []

    return [{
        'type': 'grammar',
        'original': text,
        'suggested': corrected,
        'message': "T5 grammar correction"
    }]

def correct_text(text):
    try:
        print("[DEBUG] Input text:", text)
        spelling = correct_spelling(text)
        print("[DEBUG] Spelling corrections:", spelling)

        try:
            grammar = correct_grammar(text)
        except Exception as e:
            print("[ERROR] Grammar check failed:", e)
            traceback.print_exc()
            grammar = []

        morphology = correct_morphology(text)
        print("[DEBUG] Morphology corrections:", morphology)

        try:
            gpt = correct_gpt(text)
        except Exception as e:
            print("[ERROR] GPT correction failed:", e)
            traceback.print_exc()
            gpt = []

        corrections = spelling + grammar + morphology + gpt
        print("[DEBUG] Total corrections:", corrections)

        corrected_text = text
        for c in corrections:
            o = c.get('original', '').strip()
            s = c.get('suggested', '').strip()
            if o and s and o != s:
                corrected_text = re.sub(r'\b' + re.escape(o) + r'\b', s, corrected_text)

        print("[DEBUG] Corrected text:", corrected_text)
        return corrected_text, corrections

    except Exception as e:
        print("[FATAL] Unhandled error in correct_text:", e)
        traceback.print_exc()
        return text, []

