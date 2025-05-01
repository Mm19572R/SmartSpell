import spacy
import lemminflect
nlp = spacy.load("en_core_web_sm")

def analyze_grammar(text: str) -> dict:
    doc = nlp(text)
    issues = []

    for sent in doc.sents:
        has_subject = False
        subject = None
        verb = None
        aux_verb = None
        
        # First pass to find auxiliaries and key components
        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass"):
                has_subject = True
                subject = token
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                verb = token
            # Track auxiliary verbs
            if token.dep_ == "aux" and token.pos_ == "AUX":
                aux_verb = token

        if verb and not has_subject:
            issues.append("Missing subject for the main verb.")

        # Passive voice detection
        if any(tok.dep_ == "auxpass" for tok in sent):
            issues.append("Possible passive voice; consider active voice.")

        # Third-person singular subject-verb agreement
        if subject and subject.text.lower() in ["she", "he", "it"] or (subject and (subject.tag_ == "NN" or subject.morph.get("Number") == ["Sing"])):
            # Case 1: "She don't like chocolate" -> "She doesn't like chocolate"
            if aux_verb and aux_verb.text.lower() in ["don't", "dont", "do"]:
                issues.append(
                    f"Subject-verb agreement: '{subject.text} doesn't {verb.lemma_}' instead of '{subject.text} {aux_verb.text} {verb.text}'."
                )
            # Case 2: No auxiliary but incorrect main verb form: "She like chocolate" -> "She likes chocolate"
            elif not aux_verb and verb and verb.tag_ not in ["VBZ", "VBD", "VBN", "MD"]:
                corrected_verb = verb._.inflect("VBZ") or verb.text + "s"
                issues.append(
                    f"Subject-verb agreement: '{subject.text} {corrected_verb}' instead of '{subject.text} {verb.text}'."
                )

    # Repeated words
    for a, b in zip(doc, doc[1:]):
        if a.text.lower() == b.text.lower():
            issues.append(f"Repeated word: '{a.text}'.")
            break

    # Missing end punctuation
    if text and text[-1] not in ".!?":
        issues.append("Sentence missing end punctuation.")

    return {"issues": issues}