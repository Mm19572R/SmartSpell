import spacy
import logging
from typing import List, Dict, Tuple, Optional
from lemminflect import getInflection
from spacy.tokens import Token

# Configure logging
logging.basicConfig(level=logging.DEBUG)
nlp = spacy.load("en_core_web_sm")

# Modify the `set_extension` call to avoid duplicate registration
Token.set_extension("inflect", method=lambda token, tag: getInflection(token.text, tag=tag)[0] if getInflection(token.text, tag=tag) else None, force=True)

class GrammarAnalyzer:
    def __init__(self):
        self.past_tense_markers = ['yesterday', 'last', 'ago', 'before', 'previously']
        self.present_tense_markers = ['today', 'now', 'currently', 'always', 'usually']
        
        # Common irregular verbs and their forms
        self.irregular_verbs = {
            'go': {'past': 'went', 'past_participle': 'gone'},
            'buy': {'past': 'bought', 'past_participle': 'bought'},
            'eat': {'past': 'ate', 'past_participle': 'eaten'},
            'write': {'past': 'wrote', 'past_participle': 'written'},
            'do': {'past': 'did', 'past_participle': 'done'},
            'have': {'past': 'had', 'past_participle': 'had'},
            'make': {'past': 'made', 'past_participle': 'made'},
            'see': {'past': 'saw', 'past_participle': 'seen'},
            'take': {'past': 'took', 'past_participle': 'taken'},
        }

        # Common word corrections
        self.word_corrections = {
            'alot': 'a lot',
            'definately': 'definitely',
            'recieve': 'receive',
            'seperate': 'separate',
            'occured': 'occurred',
            'untill': 'until',
            'wich': 'which'
        }

        # Add common word corrections for numbers and time units
        self.word_corrections.update({
            'tree': 'three',
            'week': 'weeks',
            'month': 'months',
            'day': 'days'
        })

    def check_sentence_boundaries(self, sent) -> Optional[str]:
        """Check sentence start/end formatting."""
        text = sent.text.strip()
        if text and text[0].islower():
            return f"Capitalization: '{text[0].upper() + text[1:]}' instead of '{text}'"
        if text and not text.endswith(('.', '!', '?')):
            return f"Missing end punctuation: '{text}.' instead of '{text}'"
        return None

    def check_subject_verb_agreement(self, subject, verb) -> Optional[str]:
        """Check subject-verb agreement."""
        if subject.text.lower() in ["he", "she", "it"]:
            if verb.tag_ in ["VB", "VBP"]:  # Base form or non-3rd person
                correct_form = verb._.inflect("VBZ") or f"{verb.text}s"
                return f"Subject-verb agreement: '{subject.text} {correct_form}' instead of '{subject.text} {verb.text}'"
        
        # Handle plural subjects
        elif subject.morph.get("Number") == ["Plur"]:
            if verb.tag_ == "VBZ":  # Third person singular with plural subject
                base_form = verb.lemma_
                return f"Subject-verb agreement: '{subject.text} {base_form}' instead of '{subject.text} {verb.text}'"
        
        return None

    def check_tense_consistency(self, sent, verb) -> Optional[str]:
        """Check if verb tense matches time context."""
        text_lower = sent.text.lower()
        
        # Check past tense markers
        if any(marker in text_lower for marker in self.past_tense_markers):
            verb_text = verb.text.lower()
            
            # First check if it's an irregular verb
            if verb.lemma_ in self.irregular_verbs:
                if verb_text != self.irregular_verbs[verb.lemma_]['past']:
                    correct_form = self.irregular_verbs[verb.lemma_]['past']
                    marker = next(m for m in self.past_tense_markers if m in text_lower)
                    return f"Tense consistency: Use '{correct_form}' instead of '{verb.text}' with '{marker}'"
            
            # Then handle regular verbs
            elif verb.tag_ in ["VB", "VBP", "VBZ"]:
                past_form = verb._.inflect("VBD") or f"{verb.text}ed"
                marker = next(m for m in self.past_tense_markers if m in text_lower)
                return f"Tense consistency: Use '{past_form}' instead of '{verb.text}' with '{marker}'"
        
        return None

    def check_missing_auxiliary(self, subject, verb) -> Optional[str]:
        """Check for missing auxiliary verbs."""
        if verb.tag_ == "VBG":  # Present participle (-ing form)
            aux = "am" if subject.text.lower() == "i" else (
                "are" if subject.text.lower() in ["you", "we", "they"] else "is"
            )
            return f"Missing auxiliary verb: '{subject.text} {aux} {verb.text}' instead of '{subject.text} {verb.text}'"
        return None

    def check_auxiliary_verb_continuous(self, sent) -> Optional[str]:
        """Check for missing auxiliary verbs in continuous tenses (e.g., 'I waiting' -> 'I am waiting')."""
        for token in sent:
            if token.text.lower() == "i" and token.head.text.lower() == "waiting":
                if not any(aux.text.lower() in ["am", "is", "are", "was", "were", "have", "been"] for aux in token.head.children):
                    return "Missing auxiliary verb: Use 'I am waiting' instead of 'I waiting'"
        return None

    def check_word_choice(self, token) -> Optional[str]:
        """Check for common word usage errors."""
        if token.text.lower() in self.word_corrections:
            correction = self.word_corrections[token.text.lower()]
            if isinstance(correction, dict):
                options = [f"'{k}' ({v})" for k, v in correction.items()]
                return f"Word choice: Consider {', '.join(options)} instead of '{token.text}'"
            else:
                return f"Word form: '{correction}' instead of '{token.text}'"
        return None

    def check_incorrect_past_forms(self, token) -> Optional[str]:
        """Check for incorrectly formed past tense verbs."""
        if token.text.lower().endswith('ed') and token.lemma_ in self.irregular_verbs:
            correct = self.irregular_verbs[token.lemma_]['past']
            return f"Incorrect verb form: Use '{correct}' instead of '{token.text}'"
        
        # Check for incorrect forms like "buyed"
        if token.text.lower() == "buyed":
            return f"Incorrect verb form: Use 'bought' instead of '{token.text}'"
        
        return None

    def check_article_usage(self, token, next_token) -> Optional[str]:
        """Check for incorrect article usage (e.g., 'a apple' -> 'an apple')."""
        if token.text.lower() == "a" and next_token.text[0].lower() in "aeiou":
            return f"Article usage: Use 'an' instead of 'a' before '{next_token.text}'"
        if token.text.lower() == "an" and next_token.text[0].lower() not in "aeiou":
            return f"Article usage: Use 'a' instead of 'an' before '{next_token.text}'"
        return None

    def check_numerical_context(self, sent) -> Optional[str]:
        """Check for numerical and temporal inconsistencies (e.g., 'tree weeks' -> 'three weeks')."""
        words = sent.text.split()
        for i, word in enumerate(words[:-1]):
            if word.lower() == "tree" and words[i + 1].lower() in ["week", "weeks", "month", "months"]:
                return f"Numerical context: Did you mean 'three {words[i + 1]}' instead of 'tree {words[i + 1]}'?"
        return None

def analyze_grammar(text: str) -> dict:
    """Analyze grammar in the text."""
    doc = nlp(text)
    issues = []
    analyzer = GrammarAnalyzer()

    # Process each sentence
    for sent in doc.sents:
        # Check sentence formatting
        boundary_error = analyzer.check_sentence_boundaries(sent)
        if boundary_error:
            issues.append(boundary_error)

        # Track sentence components
        subjects = []
        verbs = []
        has_past_marker = any(marker in sent.text.lower() for marker in analyzer.past_tense_markers)
        
        # First pass: collect subjects and verbs
        for token in sent:
            # Word choice checks
            word_error = analyzer.check_word_choice(token)
            if word_error:
                issues.append(word_error)

            # Collect subjects and verbs
            if token.dep_ in ["nsubj", "nsubjpass"]:
                subjects.append(token)
            if token.pos_ == "VERB":
                verbs.append(token)
                # If we have a past tense marker, check verb tense immediately
                if has_past_marker:
                    tense_error = analyzer.check_tense_consistency(sent, token)
                    if tense_error:
                        issues.append(tense_error)
                # Check for incorrect past tense forms
                error = analyzer.check_incorrect_past_forms(token)
                if error:
                    issues.append(error)

        # Second pass: check verb relationships
        for subject in subjects:
            for verb in verbs:
                if verb.head == subject.head or verb == subject.head:
                    # Agreement check
                    agreement_error = analyzer.check_subject_verb_agreement(subject, verb)
                    if agreement_error:
                        issues.append(agreement_error)
                    
                    # Auxiliary verb check
                    aux_error = analyzer.check_missing_auxiliary(subject, verb)
                    if aux_error:
                        issues.append(aux_error)
                    
                    # Tense consistency check
                    tense_error = analyzer.check_tense_consistency(sent, verb)
                    if tense_error:
                        issues.append(tense_error)

        # Add article usage checks in the grammar analysis loop
        for i, token in enumerate(sent[:-1]):
            next_token = sent[i + 1]
            article_error = analyzer.check_article_usage(token, next_token)
            if article_error:
                issues.append(article_error)

        # Add numerical context checks in the grammar analysis loop
        numerical_error = analyzer.check_numerical_context(sent)
        if numerical_error:
            issues.append(numerical_error)

        # Refine auxiliary verb checks for continuous tenses in the grammar analysis loop
        aux_continuous_error = analyzer.check_auxiliary_verb_continuous(sent)
        if aux_continuous_error:
            issues.append(aux_continuous_error)

        # Check for missing end punctuation
        if not str(sent).strip().endswith(('.', '!', '?')):
            issues.append("Missing end punctuation in sentence.")

    return {"issues": issues}