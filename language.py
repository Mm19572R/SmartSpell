from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def identify_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"
