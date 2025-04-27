import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from spellchecker import SpellChecker

class SpellingCorrector:
    def __init__(self, excel_path: str):
        self.vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.model = MultinomialNB()
        self.train(excel_path)
        self.spell = SpellChecker()  # Real dictionary fallback

    def train(self, excel_path: str):
        df = pd.read_excel(excel_path)
        if 'wrong' not in df.columns or 'correct' not in df.columns:
            raise ValueError("Excel file must contain 'wrong' and 'correct' columns")
        X = self.vectorizer.fit_transform(df['wrong'])
        y = df['correct']
        self.model.fit(X, y)

    def correct_word(self, word: str) -> str:
        word_lower = word.lower()
        if word_lower in self.spell.word_frequency:
            return word  # If it's a real English word, keep it
        
        try:
            X_test = self.vectorizer.transform([word])
            prediction = self.model.predict(X_test)[0]
            if prediction != word:
                return prediction
        except Exception:
            pass

        corrected = self.spell.correction(word)
        return corrected if corrected else word

    def correct_spelling(self, text: str) -> str:
        words = text.split()
        corrected_words = [self.correct_word(word) for word in words]
        return ' '.join(corrected_words)
