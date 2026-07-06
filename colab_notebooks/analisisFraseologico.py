import spacy
import unicodedata

# Cargar modelo en español
try:
    nlp = spacy.load("es_core_news_sm")
except:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")

def satira(text, satirical_words):
    doc = nlp(text)
    text = text.lower()
    
    # Normalize the text to remove accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) != 'Mn')

    # Remove punctuation
    text = ''.join(c for c in text if not c in '"#$%&\'()*+-/:<=>@[\\]^_`{|}~')

    satire_words_count = {}
    total_satire_words = 0
    for phrase in satirical_words:
        count = text.count(phrase.lower())
        satire_words_count[phrase] = count
        total_satire_words += count

    word_count = len([token.text for token in doc if not token.is_punct])
    satire_words_density = total_satire_words / word_count if word_count else 0

    result = {phrase: count for phrase, count in satire_words_count.items()}
    result["total_satire_words"] = total_satire_words
    result["satire_words_density"] = satire_words_density
    return result
