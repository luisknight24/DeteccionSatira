import nltk

def download_nltk_resources():
    print("[NLTK] Iniciando descarga de recursos...")

    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'vader_lexicon': 'sentiment/vader_lexicon'
    }
    
    for name, path in resources.items():
        print(f"[NLTK] Verificando '{name}'...")
        try:
            nltk.data.find(path)
            print(f"[NLTK] Recurso '{name}' ya esta instalado")
        except LookupError:
            print(f"[NLTK] Descargando recurso '{name}'...")
            nltk.download(name)
            print(f"[NLTK] Recurso '{name}' descargado correctamente")

if __name__ == "__main__":
    print("[NLTK] Ejecutando script")
    download_nltk_resources()