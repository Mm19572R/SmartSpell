from bert_corrector import BertCorrector

bert = BertCorrector()

def generate_alternatives(text: str, suggestions: list[dict], top_k=3):
    alts = [text]
    for s in suggestions:
        if s["type"] != "spelling":
            continue
        mask = bert.mask_token
        masked = text.replace(s["original"], mask, 1)
        preds = bert.get_suggestions(masked, s["original"], top_k=top_k)
        for p in preds:
            alts.append(masked.replace(mask, p["word"]))
    return alts
