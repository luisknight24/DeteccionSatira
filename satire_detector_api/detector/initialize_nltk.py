import nltk

def download_nltk_resources():
    print("🟡 Iniciando descarga de recursos NLTK...")  # DEBUG

    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'vader_lexicon': 'sentiment/vader_lexicon'
    }
    
    for name, path in resources.items():
        print(f"🔍 Verificando '{name}'...")
        try:
            nltk.data.find(path)
            print(f"✅ Recurso '{name}' ya está instalado")
        except LookupError:
            print(f"⬇️ Descargando recurso '{name}'...")
            nltk.download(name)
            print(f"✅ Recurso '{name}' descargado correctamente")

if __name__ == "__main__":
    print("🚀 Ejecutando script NLTK")  # DEBUG
    download_nltk_resources()
