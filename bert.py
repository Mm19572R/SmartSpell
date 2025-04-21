
import torch
# Import BERT tokenizer and masked language model from the transformers library
from transformers import BertTokenizer, BertForMaskedLM
# Import NumPy for numerical operations
import numpy as np
# Import regular expressions module for text pattern matching
import re
# Import typing for type annotations
from typing import List, Dict, Tuple

# Define the main class for BERT-based text correction
class BertCorrector:
    """
    Context-aware text correction using BERT's masked language modeling capabilities.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the BERT corrector with a pre-trained model

        Args:
            model_name: Name of the pre-trained BERT model to use
        """
        print(f"Loading BERT model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def get_suggestions(self, text: str, word: str, top_k: int = 5) -> List[Dict]:
        word_pattern = re.compile(r'\b' + re.escape(word) + r'\b')
        matches = list(word_pattern.finditer(text))
        if not matches:
            return []

        suggestions = []
        for match in matches:
            start, end = match.span()
            masked_text = text[:start] + self.mask_token + text[end:]
            with torch.no_grad():
                inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                if len(mask_token_index) == 0:
                    continue
                mask_token_index = mask_token_index.item()
                predictions = outputs.logits[0, mask_token_index].softmax(dim=0)
                top_k_scores, top_k_indices = torch.topk(predictions, top_k)
                for i, (score, idx) in enumerate(zip(top_k_scores, top_k_indices)):
                    token = self.tokenizer.convert_ids_to_tokens(idx.item())
                    if "##" not in token and len(token) > 1 and token.lower() != word.lower():
                        suggestions.append({"word": token, "confidence": float(score.item())})

        seen = set()
        unique_suggestions = []
        for suggestion in sorted(suggestions, key=lambda x: x["confidence"], reverse=True):
            if suggestion["word"] not in seen and len(unique_suggestions) < top_k:
                seen.add(suggestion["word"])
                unique_suggestions.append(suggestion)

        return unique_suggestions

    def detect_errors(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        words = re.findall(r'\b\w+\b', text)
        errors = []
        for word in words:
            if len(word) <= 2:
                continue
            word_pattern = re.compile(r'\b' + re.escape(word) + r'\b')
            match = word_pattern.search(text)
            if not match:
                continue
            start, end = match.span()
            masked_text = text[:start] + self.mask_token + text[end:]
            with torch.no_grad():
                inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                if len(mask_token_index) == 0:
                    continue
                mask_token_index = mask_token_index.item()
                predictions = outputs.logits[0, mask_token_index].softmax(dim=0)
                top_score, top_index = torch.max(predictions, dim=0)
                top_word = self.tokenizer.convert_ids_to_tokens(top_index.item())
                if top_word.lower() != word.lower() and top_score.item() > confidence_threshold:
                    errors.append({
                        "original": word,
                        "suggested": top_word,
                        "confidence": float(top_score.item()),
                        "position": (start, end)
                    })

        return {"text": text, "errors": errors}

    def correct_text(self, text: str) -> Dict:
        result = self.detect_errors(text)
        corrected_text = text
        for error in sorted(result["errors"], key=lambda x: x["position"][0], reverse=True):
            start, end = error["position"]
            corrected_text = corrected_text[:start] + error["suggested"] + corrected_text[end:]
        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "errors": result["errors"]
        }

_bert_corrector = None

def get_bert_corrector():
    global _bert_corrector
    if _bert_corrector is None:
        _bert_corrector = BertCorrector()
    return _bert_corrector

def get_smart_suggestions(text: str, word: str, top_k: int = 5) -> List[Dict]:
    corrector = get_bert_corrector()
    return corrector.get_suggestions(text, word, top_k)

def analyze_text(text: str) -> Dict:
    corrector = get_bert_corrector()
    result = corrector.detect_errors(text)
    issues = []
    for error in result["errors"]:
        issues.append(f"Consider replacing '{error['original']}' with '{error['suggested']}' ({error['confidence']:.2f} confidence)")
    return {
        "text": text,
        "issues": issues,
        "has_issues": len(issues) > 0,
        "errors": result["errors"]
    }

def auto_correct(text: str) -> Dict:
    corrector = get_bert_corrector()
    return corrector.correct_text(text)

if __name__ == "__main__":
    corrector = BertCorrector()
    test_text = "I have a problm with my computr."
    print("Testing error detection:")
    result = corrector.detect_errors(test_text)
    print(f"Original: {test_text}")
    print("Detected errors:")
    for error in result["errors"]:
        print(f"  '{error['original']}' -> '{error['suggested']}' ({error['confidence']:.2f})")

    print("\nTesting auto-correction:")
    correction = corrector.correct_text(test_text)
    print(f"Original: {correction['original_text']}")
    print(f"Corrected: {correction['corrected_text']}")
