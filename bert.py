# # Import the PyTorch library for deep learning operations
# import torch
# # Import BERT tokenizer and masked language model from the transformers library
# from transformers import BertTokenizer, BertForMaskedLM
# # Import NumPy for numerical operations
# import numpy as np
# # Import regular expressions module for text pattern matching
# import re
# # Import typing for type annotations
# from typing import List, Dict, Tuple

# # Define the main class for BERT-based text correction
# class BertCorrector:
#     """
#     Context-aware text correction using BERT's masked language modeling capabilities.
#     """
    
#     def _init_(self, model_name: str = "bert-base-uncased"):
#         """
#         Initialize the BERT corrector with a pre-trained model
        
#         Args:
#             model_name: Name of the pre-trained BERT model to use
#         """
#         # Print loading message with the model name
#         print(f"Loading BERT model: {model_name}...")
#         # Determine whether to use GPU (CUDA) or CPU for processing
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Load the pre-trained BERT tokenizer
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         # Load the pre-trained BERT masked language model and move it to the appropriate device
#         self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
#         # Set the model to evaluation mode (not training)
#         self.model.eval()
#         # Print a message confirming the model is loaded
#         print(f"Model loaded on {self.device}")
        
#         # Store the mask token (e.g., [MASK]) used by BERT
#         self.mask_token = self.tokenizer.mask_token
#         # Store the token ID corresponding to the mask token
#         self.mask_token_id = self.tokenizer.mask_token_id
    
#     def get_suggestions(self, text: str, word: str, top_k: int = 5) -> List[Dict]:
#         """
#         Get context-aware suggestions for a potentially incorrect word
        
#         Args:
#             text: The full text containing the word
#             word: The word to get suggestions for
#             top_k: Number of suggestions to return
            
#         Returns:
#             List of dictionaries with suggested replacements and their confidence scores
#         """
#         # Create a regular expression pattern to find the word with word boundaries
#         word_pattern = re.compile(r'\b' + re.escape(word) + r'\b')
#         # Find all occurrences of the word in the text
#         matches = list(word_pattern.finditer(text))
        
#         # If the word is not found in the text, return an empty list
#         if not matches:
#             return []
        
#         # Initialize an empty list to store suggestions
#         suggestions = []
        
#         # Process each occurrence of the word
#         for match in matches:
#             # Get the start and end positions of the match
#             start, end = match.span()
            
#             # Create a new text with the target word replaced by the mask token
#             masked_text = text[:start] + self.mask_token + text[end:]
            
#             # Process the masked text with BERT
#             with torch.no_grad():  # Disable gradient calculation for inference
#                 # Tokenize the masked text and convert to tensor format
#                 inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
#                 # Get model outputs (predictions)
#                 outputs = self.model(**inputs)
                
#                 # Find the position of the mask token in the input
#                 mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                
#                 # Skip if mask token is not found (shouldn't happen if masked_text was created correctly)
#                 if len(mask_token_index) == 0:
#                     continue
                    
#                 # Get the index of the mask token
#                 mask_token_index = mask_token_index.item()
                
#                 # Get probability distribution over vocabulary for the masked position
#                 predictions = outputs.logits[0, mask_token_index].softmax(dim=0)
                
#                 # Get the top k predictions with their scores
#                 top_k_scores, top_k_indices = torch.topk(predictions, top_k)
                
#                 # Convert token IDs to actual words and store with confidence scores
#                 for i, (score, idx) in enumerate(zip(top_k_scores, top_k_indices)):
#                     # Convert token ID to actual token
#                     token = self.tokenizer.convert_ids_to_tokens(idx.item())
#                     # Filter out subwords (tokens starting with ##) and the original word
#                     if "##" not in token and len(token) > 1 and token.lower() != word.lower():
#                         # Add valid suggestion to the list
#                         suggestions.append({
#                             "word": token,
#                             "confidence": float(score.item())
#                         })
        
#         # Remove duplicates and get the top k unique suggestions
#         seen = set()  # Set to track words we've already seen
#         unique_suggestions = []  # List to store unique suggestions
#         # Sort suggestions by confidence (highest first) and process
#         for suggestion in sorted(suggestions, key=lambda x: x["confidence"], reverse=True):
#             # Only add words we haven't seen before
#             if suggestion["word"] not in seen and len(unique_suggestions) < top_k:
#                 seen.add(suggestion["word"])
#                 unique_suggestions.append(suggestion)
        
#         # Return the list of unique suggestions
#         return unique_suggestions
    
#     def detect_errors(self, text: str, confidence_threshold: float = 0.7) -> Dict:
#         """
#         Detect potential errors in a text by checking if BERT would predict different words
        
#         Args:
#             text: The text to analyze
#             confidence_threshold: Minimum confidence for a suggestion
            
#         Returns:
#             Dictionary with the original text and identified potential errors
#         """
#         # Extract all words from the text
#         words = re.findall(r'\b\w+\b', text)
#         # Initialize an empty list to store detected errors
#         errors = []
        
#         # Process each word in the text
#         for word in words:
#             # Skip very short words (unlikely to be errors or meaningful)
#             if len(word) <= 2:
#                 continue
                
#             # Create a regular expression pattern to find the exact word
#             word_pattern = re.compile(r'\b' + re.escape(word) + r'\b')
#             # Find the first occurrence of the word in the text
#             match = word_pattern.search(text)
            
#             # Skip if the word can't be found (shouldn't happen if words were extracted from text)
#             if not match:
#                 continue
                
#             # Get the start and end positions of the word
#             start, end = match.span()
#             # Create a masked version of the text with this word replaced by mask token
#             masked_text = text[:start] + self.mask_token + text[end:]
            
#             # Process with BERT to get prediction for the masked position
#             with torch.no_grad():  # Disable gradient calculation for inference
#                 # Tokenize the masked text and convert to tensor format
#                 inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
#                 # Get model outputs (predictions)
#                 outputs = self.model(**inputs)
                
#                 # Find the position of the mask token in the input
#                 mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                
#                 # Skip if mask token is not found
#                 if len(mask_token_index) == 0:
#                     continue
                    
#                 # Get the index of the mask token
#                 mask_token_index = mask_token_index.item()
                
#                 # Get probability distribution over vocabulary for the masked position
#                 predictions = outputs.logits[0, mask_token_index].softmax(dim=0)
#                 # Get the most likely prediction and its confidence score
#                 top_score, top_index = torch.max(predictions, dim=0)
#                 # Convert token ID to actual token
#                 top_word = self.tokenizer.convert_ids_to_tokens(top_index.item())
                
#                 # If BERT suggests a different word with high confidence, consider it an error
#                 if top_word.lower() != word.lower() and top_score.item() > confidence_threshold:
#                     # Add to the errors list with details
#                     errors.append({
#                         "original": word,
#                         "suggested": top_word,
#                         "confidence": float(top_score.item()),
#                         "position": (start, end)
#                     })
        
#         # Return a dictionary with the original text and list of errors
#         return {
#             "text": text,
#             "errors": errors
#         }
    
#     def correct_text(self, text: str) -> Dict:
#         """
#         Automatically correct text based on BERT's predictions
        
#         Args:
#             text: The text to correct
            
#         Returns:
#             Dictionary with original and corrected text
#         """
#         # First detect all errors in the text
#         result = self.detect_errors(text)
#         # Initialize the corrected text as the original text
#         corrected_text = text
        
#         # Apply corrections from end to beginning to maintain positions
#         # (If we start from the beginning, later positions would shift)
#         for error in sorted(result["errors"], key=lambda x: x["position"][0], reverse=True):
#             # Get the start and end positions of the error
#             start, end = error["position"]
#             # Replace the original word with the suggested correction
#             corrected_text = corrected_text[:start] + error["suggested"] + corrected_text[end:]
        
#         # Return a dictionary with both original and corrected text
#         return {
#             "original_text": text,
#             "corrected_text": corrected_text,
#             "errors": result["errors"]
#         }

# # Global singleton instance to avoid reloading the model multiple times
# _bert_corrector = None

# def get_bert_corrector():
#     """
#     Get or initialize the BERT corrector
#     This is a singleton pattern - only one instance will be created
#     """
#     global _bert_corrector
#     # If the corrector hasn't been initialized yet, create it
#     if _bert_corrector is None:
#         _bert_corrector = BertCorrector()
#     # Return the existing instance
#     return _bert_corrector

# def get_smart_suggestions(text: str, word: str, top_k: int = 5) -> List[Dict]:
#     """
#     Get context-aware suggestions for a word
    
#     Args:
#         text: The complete text
#         word: The potentially incorrect word
#         top_k: Number of suggestions to return
        
#     Returns:
#         List of suggestions with confidence scores
#     """
#     # Get the BERT corrector instance
#     corrector = get_bert_corrector()
#     # Get suggestions for the word in context
#     return corrector.get_suggestions(text, word, top_k)

# def analyze_text(text: str) -> Dict:
#     """
#     Analyze text for potential errors
    
#     Args:
#         text: The text to analyze
        
#     Returns:
#         Dictionary with analysis results
#     """
#     # Get the BERT corrector instance
#     corrector = get_bert_corrector()
#     # Detect errors in the text
#     result = corrector.detect_errors(text)
    
#     # Format the results into a structure similar to analyze_grammar for consistency
#     issues = []
#     # For each detected error, create a human-readable issue message
#     for error in result["errors"]:
#         issues.append(f"Consider replacing '{error['original']}' with '{error['suggested']}' ({error['confidence']:.2f} confidence)")
    
#     # Return a dictionary with the structured results
#     return {
#         "text": text,
#         "issues": issues,
#         "has_issues": len(issues) > 0,
#         "errors": result["errors"]  # Include the detailed error information
#     }

# def auto_correct(text: str) -> Dict:
#     """
#     Automatically correct text
    
#     Args:
#         text: The text to correct
        
#     Returns:
#         Dictionary with original and corrected text
#     """
#     # Get the BERT corrector instance
#     corrector = get_bert_corrector()
#     # Apply automatic correction to the text
#     return corrector.correct_text(text)

# # Test function - only runs when script is executed directly (not imported)
# if _name_ == "_main_":
#     # Create a BERT corrector instance
#     corrector = BertCorrector()
    
#     # Test text with intentional spelling errors
#     test_text = "I have a problm with my computr."
    
#     # Test error detection
#     print("Testing error detection:")
#     # Detect errors in the test text
#     result = corrector.detect_errors(test_text)
#     # Print the original text
#     print(f"Original: {test_text}")
#     # Print the detected errors
#     print("Detected errors:")
#     for error in result["errors"]:
#         print(f"  '{error['original']}' -> '{error['suggested']}' ({error['confidence']:.2f})")
    
#     # Test automatic correction
#     print("\nTesting auto-correction:")
#     # Apply correction to the test text
#     correction = corrector.correct_text(test_text)
#     # Print original and corrected texts
#     print(f"Original: {correction['original_text']}")
#     print(f"Corrected: {correction['corrected_text']}")

# Import the PyTorch library for deep learning operations
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
