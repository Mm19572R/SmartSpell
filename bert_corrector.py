import torch
import re
from transformers import BertTokenizer, BertForMaskedLM
from spellchecker import SpellChecker
from typing import Dict

class BertCorrector:
    def __init__(self, model_name: str = "bert-base-uncased"):
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        if self.device.type == 'cuda':
            self.model = self.model.half()
        self.model.eval()

        # Fallback spellchecker
        self.spell = SpellChecker()
        # Add common contraction
        self.spell.word_frequency.add("i'm")

        # Mask info
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def correct_text(self, text: str) -> Dict:
        corrected = text
        corrections = []

        # Find all words
        for word in re.findall(r"\b\w+'\w+|\b\w+\b", text):  # includes contractions
            # Skip very short or proper nouns
            if len(word) <= 1 or (word[0].isupper() and word.lower() != word):
                continue

            # 1) SpellChecker fallback
            sc = self.spell.correction(word)
            if sc and sc.lower() != word.lower():
                corrected = re.sub(rf"\b{re.escape(word)}\b", sc, corrected, count=1)
                corrections.append({"original": word, "suggested": sc})
                continue

            # 2) BERT masked‐LM for context
            masked = re.sub(rf"\b{re.escape(word)}\b", self.mask_token, corrected, count=1)
            inputs = self.tokenizer(masked, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            # Locate mask
            idxs = (inputs.input_ids[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
            if not len(idxs):
                continue
            i = idxs.item()
            logits = out.logits[0, i].softmax(dim=0)
            top = torch.argmax(logits).item()
            pred = self.tokenizer.convert_ids_to_tokens(top).lstrip("##")

            # Only accept if high confidence and actually different
            if pred.lower() != word.lower() and logits[top] > 0.75 and len(pred) > 1:
                corrected = re.sub(rf"\b{re.escape(word)}\b", pred, corrected, count=1)
                corrections.append({"original": word, "suggested": pred})

        return {
            "original_text": text,
            "corrected_text": corrected,
            "corrections": corrections
        }

    def complete_sentence(self, partial: str, max_tokens: int = 5) -> str:
        sent = partial.strip()
        if sent.endswith((".", "!", "?")):
            return sent
        for _ in range(max_tokens):
            masked = sent + " " + self.mask_token
            inputs = self.tokenizer(masked, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            idxs = (inputs.input_ids[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
            if not len(idxs):
                break
            i = idxs.item()
            logits = out.logits[0, i].softmax(dim=0)
            top = torch.argmax(logits).item()
            pred = self.tokenizer.convert_ids_to_tokens(top).lstrip("##")
            if pred in [".", "!", "?"]:
                sent += pred
                break
            sent += " " + pred
        return sent
