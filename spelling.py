import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class SpellingCorrector:
    def __init__(self, excel_path: str):
        self.vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.model = MultinomialNB()
        self.train(excel_path)

    def train(self, excel_path: str):
        df = pd.read_excel(excel_path)
        if 'wrong' not in df.columns or 'correct' not in df.columns:
            raise ValueError("Excel file must contain 'wrong' and 'correct' columns")
        X = self.vectorizer.fit_transform(df['wrong'])
        y = df['correct']
        self.model.fit(X, y)

    def correct_word(self, word: str) -> str:
        try:
            X_test = self.vectorizer.transform([word])
            prediction = self.model.predict(X_test)
            return prediction[0]
        except:
            return word

    def correct_spelling(self, text: str) -> str:
        words = text.split()
        corrected_words = [self.correct_word(word) for word in words]
        return ' '.join(corrected_words)
