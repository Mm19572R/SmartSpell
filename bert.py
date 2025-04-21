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

    #############################################################
    # SECTION: Grammar Correction Beyond Basic Rules
    #############################################################
    
    def detect_grammar_issues(self, text: str, confidence_threshold: float = 0.6) -> List[Dict]:
        """
        Detects grammar issues beyond basic rules by analyzing sentence structure with BERT.
        
        Args:
            text: Input text to analyze
            confidence_threshold: Confidence threshold for suggestions
            
        Returns:
            List of grammar issues with suggested corrections
        """
        # Split text into sentences for processing
        sentences = re.split(r'(?<=[.!?])\s+', text)
        issues = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Process each word in the sentence to find grammatical inconsistencies
            tokens = self.tokenizer.tokenize(sentence)
            
            # Check for common grammar patterns that might indicate issues
            for i in range(len(tokens) - 1):
                # Skip special tokens and punctuation
                if tokens[i].startswith("##") or not tokens[i].isalpha():
                    continue
                    
                # Create a masked version to check if current word is grammatically appropriate
                original_token = tokens[i]
                masked_sentence = sentence.replace(original_token, self.mask_token, 1)
                
                # Process with BERT to get alternatives
                with torch.no_grad():
                    inputs = self.tokenizer(masked_sentence, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                    
                    if len(mask_token_index) == 0:
                        continue
                        
                    mask_token_index = mask_token_index.item()
                    predictions = outputs.logits[0, mask_token_index].softmax(dim=0)
                    
                    # Get the top prediction
                    top_score, top_index = torch.max(predictions, dim=0)
                    top_word = self.tokenizer.convert_ids_to_tokens(top_index.item())
                    
                    # Check if the model strongly suggests a different word (potential grammar issue)
                    if (top_word.lower() != original_token.lower() and 
                            top_score.item() > confidence_threshold and 
                            len(top_word) > 1):
                        
                        # Found potential grammar issue
                        word_pattern = re.compile(r'\b' + re.escape(original_token) + r'\b')
                        match = word_pattern.search(sentence)
                        
                        if match:
                            start, end = match.span()
                            issues.append({
                                "type": "grammar",
                                "original": original_token,
                                "suggested": top_word,
                                "confidence": float(top_score.item()),
                                "position": (start, end),
                                "context": sentence
                            })
        
        return issues

    def fix_grammar(self, text: str) -> Dict:
        """
        Applies grammar fixes beyond basic rules to the text.
        
        Args:
            text: Text to correct
            
        Returns:
            Dictionary with original and corrected text, plus details of fixed issues
        """
        issues = self.detect_grammar_issues(text)
        corrected_text = text
        
        # Sort issues by position in reverse order to avoid offsetting issues when replacing
        for issue in sorted(issues, key=lambda x: x["position"][0], reverse=True):
            start, end = issue["position"]
            # Replace the word with the suggestion
            corrected_text = corrected_text[:start] + issue["suggested"] + corrected_text[end:]
            
        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "fixes": issues
        }
    
    #############################################################
    # SECTION: Style Improvement Suggestions
    #############################################################
    
    def analyze_style(self, text: str) -> Dict:
        """
        Analyzes text style and provides suggestions for improvement.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with style analysis and improvement suggestions
        """
        # Define common style patterns to look for
        style_patterns = {
            "complex_phrases": [
                ("in order to", "to"),
                ("due to the fact that", "because"),
                ("in spite of the fact that", "although"),
                ("at this point in time", "now"),
                ("for the purpose of", "for"),
                ("in the event that", "if"),
                ("prior to", "before"),
                ("subsequent to", "after"),
                ("in the vicinity of", "near"),
                ("has the ability to", "can"),
                ("in a timely manner", "promptly")
            ],
            "passive_voice_indicators": [
                "was", "were", "been", "be", "being", "is", "are"
            ],
            "filler_words": [
                "very", "really", "quite", "actually", "basically", "literally",
                "just", "simply", "definitely", "certainly", "probably"
            ]
        }
        
        improvements = []
        
        # Check for complex phrases and suggest simpler alternatives
        for complex_phrase, alternative in style_patterns["complex_phrases"]:
            if complex_phrase in text.lower():
                position = text.lower().find(complex_phrase)
                improvements.append({
                    "type": "complex_phrase",
                    "original": complex_phrase,
                    "suggestion": alternative,
                    "position": (position, position + len(complex_phrase)),
                    "message": f"Consider replacing '{complex_phrase}' with '{alternative}' for clarity."
                })
        
        # Check for potential passive voice constructs
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in style_patterns["passive_voice_indicators"]:
                # Look for past participle following the auxiliary verb
                if i < len(words) - 1 and i > 0:
                    # Simple heuristic to detect passive voice: auxiliary verb + past participle
                    if words[i+1].endswith("ed") or words[i+1].endswith("en"):
                        context = " ".join(words[max(0, i-2):min(len(words), i+3)])
                        improvements.append({
                            "type": "passive_voice",
                            "context": context,
                            "message": f"Possible passive voice detected in '{context}'. Consider using active voice for more direct writing."
                        })
        
        # Check for filler words that could be eliminated
        for filler in style_patterns["filler_words"]:
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = list(re.finditer(pattern, text.lower()))
            for match in matches:
                start, end = match.span()
                improvements.append({
                    "type": "filler_word",
                    "original": match.group(),
                    "position": (start, end),
                    "message": f"Consider removing the filler word '{match.group()}' for more concise writing."
                })
        
        # Analyze sentence length variation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        # Detect very long sentences (potentially hard to read)
        for i, length in enumerate(sentence_lengths):
            if length > 25:  # Threshold for long sentences
                improvements.append({
                    "type": "long_sentence",
                    "sentence": sentences[i],
                    "length": length,
                    "message": f"Long sentence detected ({length} words). Consider breaking it into smaller sentences for readability."
                })
        
        return {
            "text": text,
            "style_improvements": improvements,
            "has_improvements": len(improvements) > 0
        }
    
    def suggest_style_improvements(self, text: str) -> Dict:
        """
        Provides a formatted list of style improvement suggestions.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with original text and style suggestions
        """
        analysis = self.analyze_style(text)
        
        # Group suggestions by type for better presentation
        grouped_suggestions = {
            "complex_phrases": [],
            "passive_voice": [],
            "filler_words": [],
            "long_sentences": []
        }
        
        for improvement in analysis["style_improvements"]:
            if improvement["type"] == "complex_phrase":
                grouped_suggestions["complex_phrases"].append(improvement)
            elif improvement["type"] == "passive_voice":
                grouped_suggestions["passive_voice"].append(improvement)
            elif improvement["type"] == "filler_word":
                grouped_suggestions["filler_words"].append(improvement)
            elif improvement["type"] == "long_sentence":
                grouped_suggestions["long_sentences"].append(improvement)
        
        return {
            "text": text,
            "suggestions": grouped_suggestions,
            "has_suggestions": analysis["has_improvements"]
        }
    
    #############################################################
    # SECTION: Sentence Completion/Rewriting
    #############################################################
    
    def complete_sentence(self, partial_sentence: str, max_tokens: int = 10) -> str:
        """
        Completes a partial sentence using BERT's predictive capabilities.
        
        Args:
            partial_sentence: The incomplete sentence to finish
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Completed sentence
        """
        completed_sentence = partial_sentence.strip()
        
        # Don't try to complete sentences that are already complete
        if completed_sentence.endswith(('.', '!', '?')):
            return completed_sentence
            
        current_sentence = completed_sentence
        
        # Iteratively predict the next word and append it to build the complete sentence
        for _ in range(max_tokens):
            # Add a mask token to predict the next word
            masked_text = current_sentence + " " + self.mask_token
            
            with torch.no_grad():
                inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                
                if len(mask_token_index) == 0:
                    break
                    
                mask_token_index = mask_token_index.item()
                predictions = outputs.logits[0, mask_token_index].softmax(dim=0)
                
                # Get the top prediction
                top_score, top_index = torch.max(predictions, dim=0)
                next_word = self.tokenizer.convert_ids_to_tokens(top_index.item())
                
                # Clean up the predicted token (remove ##)
                if next_word.startswith("##"):
                    next_word = next_word[2:]
                
                # Add space before appending if it's not a punctuation
                if next_word.isalpha():
                    current_sentence += " " + next_word
                else:
                    current_sentence += next_word
                
                # Stop if we've reached the end of a sentence
                if next_word in ['.', '!', '?']:
                    break
                
                # Also stop if a period appears within the token (common for abbreviations)
                if '.' in next_word:
                    break
        
        return current_sentence
    
    def rewrite_sentence(self, sentence: str, target_style: str = "clear") -> str:
        """
        Rewrites a sentence in a target style.
        
        Args:
            sentence: The sentence to rewrite
            target_style: The desired style ("clear", "formal", "simple")
            
        Returns:
            Rewritten sentence
        """
        # Define style markers to prepend to the sentence for guiding BERT
        style_markers = {
            "clear": "Clearly stated, ",
            "formal": "In formal language, ",
            "simple": "Simply put, "
        }
        
        # Add the style marker at the beginning as context
        marker = style_markers.get(target_style, "")
        
        # Replace each word with a mask and get BERT's prediction for the entire sentence
        words = re.findall(r'\b\w+\b', sentence)
        masked_sentence = sentence
        
        # We don't want to mask every single word as that would lose too much context
        # Instead, mask about 1/3 of the words to maintain context while allowing style changes
        words_to_mask = max(1, len(words) // 3)
        mask_indices = np.random.choice(len(words), words_to_mask, replace=False)
        
        # Start with the original sentence
        rewritten = sentence
        
        # Iterate through selected words and mask them one by one for rewriting
        for word_idx in mask_indices:
            word = words[word_idx]
            word_pattern = re.compile(r'\b' + re.escape(word) + r'\b')
            match = word_pattern.search(rewritten)
            
            if not match:
                continue
                
            start, end = match.span()
            
            # Create a masked version of the current rewritten text
            masked_text = rewritten[:start] + self.mask_token + rewritten[end:]
            
            # Prepend the style marker for context
            guided_masked_text = marker + masked_text
            
            with torch.no_grad():
                inputs = self.tokenizer(guided_masked_text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                
                # Find the mask token index
                mask_token_indices = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                if len(mask_token_indices) == 0:
                    continue
                    
                # Use the first mask token index
                mask_idx = mask_token_indices[0].item()
                predictions = outputs.logits[0, mask_idx].softmax(dim=0)
                
                # Get the top prediction
                top_score, top_index = torch.max(predictions, dim=0)
                replacement = self.tokenizer.convert_ids_to_tokens(top_index.item())
                
                # Clean up the predicted token
                if replacement.startswith("##"):
                    replacement = replacement[2:]
                
                # Update the sentence with the new word
                rewritten = rewritten[:start] + replacement + rewritten[end:]
        
        # Clean up any artifacts and return the rewritten sentence
        return rewritten.strip()
    
   #############################################################
    # SECTION: Context-Aware Error Correction (FIXED)
    #############################################################
    
    def correct_homonym_errors(self, text: str) -> Dict:
        """
        Corrects commonly confused words (homonyms) based on context.
        
        Args:
            text: Text to check for homonym errors
            
        Returns:
            Dictionary with original and corrected text, plus details of corrections
        """
        # Define common homonym pairs to check
        homonym_pairs = [
            # Format: (word1, word2)
            ("their", "there"),
            ("their", "they're"),
            ("your", "you're"),
            ("its", "it's"),
            ("affect", "effect"),
            ("accept", "except"),
            ("then", "than"),
            ("to", "too"),
            ("weather", "whether"),
            ("lose", "loose"),
            ("principal", "principle"),
            ("stationary", "stationery"),
            ("complement", "compliment"),
            ("desert", "dessert"),
            ("capital", "capitol")
        ]
        
        corrections = []
        corrected_text = text
        
        # Split text into sentences for better context handling
        sentences = re.split(r'(?<=[.!?])\s+', text)
        offset = 0
        
        for sentence in sentences:
            if not sentence.strip():
                offset += len(sentence)
                continue
                
            # Look for each potential homonym in the sentence
            for word1, word2 in homonym_pairs:
                # Check for word1
                word1_pattern = re.compile(r'\b' + re.escape(word1) + r'\b')
                matches = list(word1_pattern.finditer(sentence))
                
                for match in matches:
                    local_start, local_end = match.span()
                    # Adjust to global position in text
                    start = offset + local_start
                    end = offset + local_end
                    
                    # Create masked sentence
                    masked_sentence = sentence[:local_start] + self.mask_token + sentence[local_end:]
                    
                    # Process with BERT
                    try:
                        with torch.no_grad():
                            inputs = self.tokenizer(masked_sentence, return_tensors="pt").to(self.device)
                            outputs = self.model(**inputs)
                            mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                            
                            if len(mask_token_index) == 0:
                                continue
                                
                            mask_token_index = mask_token_index.item()
                            predictions = outputs.logits[0, mask_token_index].softmax(dim=0)
                            
                            # Get token IDs for both words in the pair
                            word1_id = self.tokenizer.encode(word1, add_special_tokens=False)[0]
                            word2_id = self.tokenizer.encode(word2, add_special_tokens=False)[0]
                            
                            # Check if model prefers word2 over word1
                            if word1_id < len(predictions) and word2_id < len(predictions):
                                word1_score = predictions[word1_id].item()
                                word2_score = predictions[word2_id].item()
                                
                                # If the model strongly prefers word2
                                if word2_score > word1_score * 1.5 and word2_score > 0.3:
                                    corrections.append({
                                        "type": "homonym",
                                        "original": word1,
                                        "correction": word2,
                                        "position": (start, end),
                                        "context": sentence,
                                        "confidence": float(word2_score)
                                    })
                    except Exception as e:
                        # If any error occurs during processing, skip this match
                        print(f"Error processing homonym {word1}: {str(e)}")
                        continue
                
                # Now check for word2
                word2_pattern = re.compile(r'\b' + re.escape(word2) + r'\b')
                matches = list(word2_pattern.finditer(sentence))
                
                for match in matches:
                    local_start, local_end = match.span()
                    # Adjust to global position
                    start = offset + local_start
                    end = offset + local_end
                    
                    # Create masked sentence
                    masked_sentence = sentence[:local_start] + self.mask_token + sentence[local_end:]
                    
                    # Process with BERT
                    try:
                        with torch.no_grad():
                            inputs = self.tokenizer(masked_sentence, return_tensors="pt").to(self.device)
                            outputs = self.model(**inputs)
                            mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                            
                            if len(mask_token_index) == 0:
                                continue
                                
                            mask_token_index = mask_token_index.item()
                            predictions = outputs.logits[0, mask_token_index].softmax(dim=0)
                            
                            # Get token IDs
                            word1_id = self.tokenizer.encode(word1, add_special_tokens=False)[0]
                            word2_id = self.tokenizer.encode(word2, add_special_tokens=False)[0]
                            
                            # Check if model prefers word1 over word2
                            if word1_id < len(predictions) and word2_id < len(predictions):
                                word1_score = predictions[word1_id].item()
                                word2_score = predictions[word2_id].item()
                                
                                if word1_score > word2_score * 1.5 and word1_score > 0.3:
                                    corrections.append({
                                        "type": "homonym",
                                        "original": word2,
                                        "correction": word1,
                                        "position": (start, end),
                                        "context": sentence,
                                        "confidence": float(word1_score)
                                    })
                    except Exception as e:
                        # If any error occurs during processing, skip this match
                        print(f"Error processing homonym {word2}: {str(e)}")
                        continue
            
            # Update offset for next sentence
            offset += len(sentence) + 1  # +1 for the space
        
        # Apply corrections from end to beginning to maintain correct positions
        for correction in sorted(corrections, key=lambda x: x["position"][0], reverse=True):
            start, end = correction["position"]
            if start < len(corrected_text) and end <= len(corrected_text):
                corrected_text = corrected_text[:start] + correction["correction"] + corrected_text[end:]
            else:
                print(f"Warning: Position ({start}, {end}) out of range for text of length {len(corrected_text)}")
        
        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "corrections": corrections,
            "has_corrections": len(corrections) > 0
        }
    
    def contextual_spell_check(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Performs context-aware spell checking by examining each word in context.
        
        Args:
            text: Text to check
            confidence_threshold: Confidence threshold for suggestions
            
        Returns:
            Dictionary with spell check results
        """
        corrections = []
        corrected_text = text
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        offset = 0
        
        for sentence in sentences:
            if not sentence.strip():
                offset += len(sentence)
                continue
                
            # Find all words in the sentence
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence)
            
            for word in words:
                # Skip proper nouns (capitalized words not at start of sentence)
                if word[0].isupper() and sentence.strip()[0] != word[0]:
                    continue
                    
                # Find the word position in this sentence
                word_pattern = re.compile(r'\b' + re.escape(word) + r'\b')
                match = word_pattern.search(sentence)
                
                if not match:
                    continue
                    
                local_start, local_end = match.span()
                # Global position in the full text
                start = offset + local_start
                end = offset + local_end
                
                # Create masked sentence
                masked_sentence = sentence[:local_start] + self.mask_token + sentence[local_end:]
                
                try:
                    # Use BERT to predict the masked word
                    with torch.no_grad():
                        inputs = self.tokenizer(masked_sentence, return_tensors="pt").to(self.device)
                        outputs = self.model(**inputs)
                        mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_token_id)[0]
                        
                        if len(mask_token_index) == 0:
                            continue
                            
                        mask_token_index = mask_token_index.item()
                        predictions = outputs.logits[0, mask_token_index].softmax(dim=0)
                        
                        # Get top 5 predictions
                        top_k = 5
                        top_scores, top_indices = torch.topk(predictions, top_k)
                        
                        found_correction = False
                        for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
                            prediction = self.tokenizer.convert_ids_to_tokens(idx.item())
                            
                            # Clean up the predicted token
                            if prediction.startswith("##"):
                                prediction = prediction[2:]
                                
                            # Skip if prediction is too short or identical to original
                            if len(prediction) < 3 or prediction.lower() == word.lower():
                                continue
                                
                            # Calculate string similarity
                            similarity = self._string_similarity(word.lower(), prediction.lower())
                            
                            # If words are similar but not identical, and prediction has high confidence
                            # Higher similarity threshold (0.7) means words need to be quite similar
                            if similarity > 0.7 and similarity < 1.0 and score.item() > confidence_threshold:
                                corrections.append({
                                    "type": "contextual_spelling",
                                    "original": word,
                                    "correction": prediction,
                                    "confidence": float(score.item()),
                                    "position": (start, end),
                                    "context": sentence,
                                    "similarity": similarity
                                })
                                found_correction = True
                                break
                        
                        # If we didn't find a very similar word with high confidence,
                        # check if the top prediction is significantly more confident
                        if not found_correction and top_scores[0].item() > 0.8:
                            top_word = self.tokenizer.convert_ids_to_tokens(top_indices[0].item())
                            if top_word.startswith("##"):
                                top_word = top_word[2:]
                                
                            # Only suggest replacement if the words are somewhat similar
                            # but the model is very confident
                            similarity = self._string_similarity(word.lower(), top_word.lower())
                            if similarity > 0.5 and similarity < 1.0:
                                corrections.append({
                                    "type": "contextual_spelling",
                                    "original": word,
                                    "correction": top_word,
                                    "confidence": float(top_scores[0].item()),
                                    "position": (start, end),
                                    "context": sentence,
                                    "similarity": similarity
                                })
                except Exception as e:
                    # If any error occurs during processing, skip this word
                    print(f"Error processing word '{word}': {str(e)}")
                    continue
            
            # Update offset for next sentence
            offset += len(sentence) + 1  # +1 for the space
        
        # Apply corrections from end to beginning
        for correction in sorted(corrections, key=lambda x: x["position"][0], reverse=True):
            start, end = correction["position"]
            if start < len(corrected_text) and end <= len(corrected_text):
                corrected_text = corrected_text[:start] + correction["correction"] + corrected_text[end:]
            else:
                print(f"Warning: Position ({start}, {end}) out of range for text of length {len(corrected_text)}")
        
        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "corrections": corrections,
            "has_corrections": len(corrections) > 0
        }
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity ratio between two strings using Levenshtein distance.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity ratio between 0 and 1
        """
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0  # Both strings empty
        return 1.0 - (distance / max_len)
    
    def comprehensive_text_check(self, text: str) -> Dict:
        """
        Performs a comprehensive check combining contextual spelling and homonym error detection.
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with comprehensive check results
        """
        # First check for homonym errors
        homonym_result = self.correct_homonym_errors(text)
        
        # Then check for contextual spelling errors on the text with homonym corrections
        spell_result = self.contextual_spell_check(homonym_result["corrected_text"])
        
        # Combine the corrections
        all_corrections = homonym_result["corrections"] + spell_result["corrections"]
        
        return {
            "original_text": text,
            "corrected_text": spell_result["corrected_text"],
            "corrections": all_corrections,
            "homonym_corrections": homonym_result["corrections"],
            "spelling_corrections": spell_result["corrections"],
            "has_corrections": len(all_corrections) > 0
        }