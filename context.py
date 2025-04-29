import spacy

nlp = spacy.load("en_core_web_sm")

def analyze_context(text: str) -> dict:
    doc = nlp(text)
    return {
        "sentences": [sent.text for sent in doc.sents],
        "pos_counts": doc.count_by(spacy.attrs.POS),
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
    }
