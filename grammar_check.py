import spacy

nlp = spacy.load("en_core_web_sm")

def analyze_grammar(sentence):
    doc = nlp(sentence)
    issues = []

    has_subject = any(token.dep_ in ("nsubj", "nsubjpass", "expl") for token in doc)
    if not has_subject:
        issues.append("Sentence may be missing a subject.")

    for token in doc:
        if token.is_oov and token.is_alpha and len(token.text) > 3:
            issues.append(f"Possible spelling mistake in: '{token.text}'")

    return {"issues": issues}
