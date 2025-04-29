import spacy
import lemminflect
nlp = spacy.load("en_core_web_sm")


nlp = spacy.load("en_core_web_sm")

def analyze_grammar(text: str) -> dict:
    doc = nlp(text)
    issues = []

    for sent in doc.sents:
        has_subject = False
        subject = None
        verb = None
        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass"):
                has_subject = True
                subject = token
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                verb = token

        if verb and not has_subject:
            issues.append("Missing subject for the main verb.")

        # Passive voice detection
        if any(tok.dep_ == "auxpass" for tok in sent):
            issues.append("Possible passive voice; consider active voice.")

        # Subject-verb agreement logic (corrected)
        if subject and verb:
            if subject.tag_ in ("PRP", "NN") and subject.morph.get("Number") == ["Sing"]:
                if verb.tag_ not in ("VBZ", "VBD", "VBN"):
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
