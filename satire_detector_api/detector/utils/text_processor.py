import re
import math
import emoji
import textstat
import unicodedata
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from .model_loader import ModelLoader
from .model_loader import BertClassifier
from nltk.corpus import stopwords
from transformers import BertTokenizerFast, AutoModel, pipeline
import torch
import threading

class TextProcessor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(TextProcessor, cls).__new__(cls, *args, **kwargs)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        # Cargar el modelo de spaCy y el analizador de sentimientos VADER
        self.nlp = spacy.load("es_core_news_sm")
        self.sia = SentimentIntensityAnalyzer()
        self.model_loader = ModelLoader()
        self.common_words_es = self.model_loader.common_words_es
        self.satirical_words = self.model_loader.satirical_words
        self.stopwords_es = set(stopwords.words('spanish'))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_lock = threading.RLock()
        self._initialized = True


    @property
    def irony_detector(self):
        if not hasattr(self, '_irony_detector'):
            self._irony_detector = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-irony",
                device=0 if torch.cuda.is_available() else -1
            )
        return self._irony_detector

    selected_features = [
        # Métricas básicas de texto
        # Ya calculado en extract_features()
            'MeanWordLen',
            'LexicalDiversity',
            'MeanSentenceLen',
            'StdevSentenceLen',
            'DocumentLen',
            'WordsPerText',
            # Métricas de oraciones
            'SentencesPerText',
            'num_words',  # Ya calculado en extract_features()
            'num_chars',
            'irony_score',
            'prop_NOUN',
            'prop_VERB',
            'prop_ADJ',
            'rhetorical_questions',
            'avg_depth',
            'Flesch Score',

            # Ironía y sátira (ya calculado en extract_features() y satira())
            'Lexical Entropy',

            'Syntactic Repetition',
            'Unusual Word Frequency',

    ]
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r"[^a-zA-Záéíóúüñ¿?¡!.,;]", " ", text)

        doc = self.nlp(text)
        text = " ".join([token.lemma_ for token in doc if token.text not in self.stopwords_es])
        return text




    def satira(self, text):
       
        doc = self.nlp(text)
        text = text.lower()
        # Normaliza y limpia el texto (quita acentos y puntuación)

    # Normalize the text to remove accents
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                    if unicodedata.category(c) != 'Mn')

    # Remove punctuation.
        text = ''.join(c for c in text if not c in '"#$%&\'()*+-/:<=>@[\\]^_`{|}~')

        satire_words_count = {}
        total_satire_words = 0
        for phrase in self.model_loader.satirical_words:  # <- Lista cargada desde tu CSV
            satire_words_count[phrase] = text.count(phrase.lower())
            total_satire_words += satire_words_count[phrase]

        word_count = len([token.text for token in doc if not token.is_punct])
        satire_words_density = total_satire_words / word_count if word_count else 0

        return {
            **satire_words_count,
            "total_satire_words": total_satire_words,
            "satire_words_density": satire_words_density
        }



    # Configuración inicial (ejecutar una vez)


    def extract_features( self, text):
        #self.irony_detector = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")
        doc = self.nlp(text)

        # 1️⃣ Longitud del texto
        num_words = len(word_tokenize(text))
        num_chars = len(text)

        # 2️⃣ Uso de puntuación irónica
        exclamations = text.count("!")
        #questions = text.count("?") #ojo con rhetorical_questions
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))  # Proporción de mayúsculas

        # 3️⃣ Conteo de palabras clave de sátira
        words = [token.text.lower() for token in doc]

        # 4️⃣ Análisis de polaridad y subjetividad
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Valores entre -1 (negativo) y 1 (positivo)
        subjectivity = blob.sentiment.subjectivity  # 0 (objetivo) a 1 (subjetivo)
        # Polaridad con VADER (otro método para comparación)
        vader_polarity = self.sia.polarity_scores(text)['compound'] #Útil para analizar textos cortos con lenguaje coloquial.

        # 5️⃣ Análisis de ironía con modelo preentrenado
        if getattr(self.model_loader, 'is_mock', False):
            irony_score = 0.5
        else:
            try:
                with self.inference_lock:
                    with torch.no_grad():
                        irony_res = self.irony_detector(text)[0]
                irony_score = irony_res['score'] if irony_res['label'] == 'irony' else 1 - irony_res['score']
            except Exception as e:
                print(f"[TextProcessor] Error en detector de ironia: {e}")
                irony_score = 0.5

        # 6️⃣ Conteo de adverbios, conectores y preguntas retóricas
        prop_ADV = sum(1 for token in doc if token.pos_ == "ADV") / num_words if num_words > 0 else 0
        prop_NOUN = sum(1 for token in doc if token.pos_ == "NOUN") / num_words if num_words > 0 else 0
        prop_VERB = sum(1 for token in doc if token.pos_ == "VERB") / num_words if num_words > 0 else 0
        prop_ADJ = sum(1 for token in doc if token.pos_ == "ADJ") / num_words if num_words > 0 else 0
        rhetorical_questions = sum(1 for sent in doc.sents if sent.text.strip().endswith("?"))

        # 7️⃣ Conteo de metáforas (simplificado con "como" o "es como")
        metaphors = len(re.findall(r'\bcomo\b|\bes como\b', text, re.IGNORECASE))
        satire_words_count = self.satira(text)  # Call the satira function


        return {
            "num_words": num_words,
            "num_chars": num_chars,
            "exclamations": exclamations,
            #"questions": questions,
            "uppercase_ratio": uppercase_ratio,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "Polaridad_VADER": vader_polarity,
            "irony_score": irony_score,
            "prop_ADV" : prop_ADV,
            "prop_NOUN" : prop_NOUN,
            "prop_VERB": prop_VERB,
            "prop_ADJ": prop_ADJ,
            "rhetorical_questions": rhetorical_questions,

        }

    def calculate_dependency_metrics(self, text):
        #nlp = spacy.load("es_core_news_sm")
     
        doc = self.nlp(text)

        total_depth = 0
        total_length = 0
        sentence_count = 0

        results = []

        for sent in doc.sents:
            depths = [token.head.i - token.i if token.head != token else 0 for token in sent]
            depth = max(depths) if depths else 0  # Profundidad máxima en la oración
            length = sum(abs(dep) for dep in depths)  # Longitud de dependencia total

            total_depth += depth
            total_length += length
            sentence_count += 1

            results.append({"sentence": sent.text, "depth": depth, "length": length})

        avg_depth = total_depth / sentence_count if sentence_count else 0
        avg_length = total_length / sentence_count if sentence_count else 0

        return {
            "avg_depth": avg_depth,
            "avg_length": avg_length
            #"sentence_metrics": results
        }
    def flesch_score(self, text, lang="es"):
        """Calcula el puntaje de Flesch."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return None  # Evita errores si el texto está vacío

        num_sentences = len(sentences)
        num_words = sum(len(s.split()) for s in sentences)
        num_syllables = sum(textstat.syllable_count(word) for word in text.split())

        asl = num_words / num_sentences  # Longitud media de oraciones
        asw = num_syllables / num_words  # Sílabas promedio por palabra

        score = 206.84 - (1.02 * asl) - (60 * asw)  #PARA TEXTO EN ESPAÑOL


        return round(score, 2)

    def lexical_entropy(self, text):
        words = text.lower().split()
        word_counts = Counter(words)
        total_words = sum(word_counts.values())

        entropy = -sum((count / total_words) * math.log2(count / total_words)
                for count in word_counts.values())
        return round(entropy, 2)
    def syntactic_pattern_repetition(self, text):
       
        doc = self.nlp(text)
        dep_patterns = [token.dep_ for token in doc]  # Ej: ['nsubj', 'ROOT', 'dobj']
        dep_counts = Counter(dep_patterns)

        repetition_index = max(dep_counts.values()) / len(dep_patterns) if dep_patterns else 0
        return round(repetition_index, 2)


    def unusual_word_frequency(self, text):
        words_in_text = set(word.lower() for word in text.split())
        common_set = self.model_loader.common_words_es  # <- Set cargado desde CREA_PalabrasComunes.txt
        unusual_words = words_in_text - common_set
        return round(len(unusual_words) / len(words_in_text), 2) if words_in_text else 0
    def analyze_text(self, text):
        return {
            "Flesch Score": self.flesch_score(text),  # Usa textstat.flesch_reading_ease()
            "Lexical Entropy": self.lexical_entropy(text),  # Basado en Counter() y math.log2
            "Syntactic Repetition": self.syntactic_pattern_repetition(text),  # Conteo de dependencias
            "Unusual Word Frequency": self.unusual_word_frequency(text)  # Compara con common_words_es
        }


    def LexicalDiversity(self, text, lst=None):
        separateWords = re.findall(r"\w+", text.lower())
        diversity = (len(set(separateWords)) / len(separateWords)) * 100 if separateWords else 0
        if lst is not None:
            lst.append(diversity)
        return diversity
    lexical_diversity_list = []
    def mean_sentence_len(self, text):
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        sentence_word_lengths = [len(sent.split()) for sent in sentences]
        return np.mean(sentence_word_lengths)
    def stdev_sentence_len(self, text):
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.0  # No hay desviación estándar con menos de 2 oraciones
        sentence_word_lengths = [len(sent.split()) for sent in sentences]
        return np.std(sentence_word_lengths)
    def sentences_per_text(self, text):
        """Número total de oraciones en el texto (equivalente a SentencesPerText)"""
        return len(sent_tokenize(text)) if text.strip() else 0
    def lexical_diversity(self, text):
        """Diversidad léxica (porcentaje de palabras únicas)"""
        words = re.findall(r"\w+", text.lower())
        return (len(set(words)) / len(words)) * 100 if words else 0

    def mean_word_len(self, text):
        """Longitud promedio de las palabras (MeanWordLen)"""
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        return np.mean([len(word) for word in words]) if words else 0

    def words_per_text(self, text):
        """Número total de palabras en el texto (WordsPerText)"""
        # Opción 1: Usando split() básico (si no necesitas limpieza compleja)
        words = re.sub(r'[^\w\s]', '', text.lower()).split()  # Elimina puntuación y divide

        # Opción 2: Usando word_tokenize de NLTK (para tokenización más precisa)
        # words = word_tokenize(re.sub(r'[^\w\s]', '', text.lower()))

        return len(words) if text.strip() else 0

    def document_len(self, text):
        """Longitud total del documento en caracteres (DocumentLen)"""
        sentences = sent_tokenize(text)  # Divide el texto en oraciones
        return sum(len(sentence) for sentence in sentences) if sentences else 0

    def calculate_features(self, text):
        """Combina todas las características ya calculadas por las funciones auxiliares"""
        # Obtener todas las métricas de las funciones existentes
        processed_text = self.preprocess_text(text)
        basic_features = self.extract_features(processed_text)
        satire_features = self.satira(processed_text)
        dependency_metrics = self.calculate_dependency_metrics(processed_text)
        text_analysis = self.analyze_text(processed_text)

        # Calcular algunas métricas adicionales que no están en las funciones anteriores
        doc = self.nlp(processed_text)
        tokens = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Combinar todos los features
        combined_features = {
            # Métricas básicas de texto
            'num_words': basic_features['num_words'],  # Ya calculado en extract_features()
            'num_chars': basic_features['num_chars'],  # Ya calculado en extract_features()
            'MeanWordLen': self.mean_word_len(processed_text),
            'LexicalDiversity': self.lexical_diversity(processed_text),
            'DocumentLen': self.document_len(processed_text),
            'WordsPerText': self.words_per_text(processed_text),

            # Métricas de oraciones
            'SentencesPerText': self.sentences_per_text(processed_text),
            'MeanSentenceLen': self.mean_sentence_len(processed_text),
            'StdevSentenceLen': self.stdev_sentence_len(processed_text),

            # POS tagging y estructura (ya calculado en extract_features())
            'prop_NOUN': basic_features['prop_NOUN'],
            'prop_VERB': basic_features['prop_VERB'],
            'prop_ADJ': basic_features['prop_ADJ'],
            'rhetorical_questions': basic_features['rhetorical_questions'],
            'avg_depth': dependency_metrics['avg_depth'],

            # Legibilidad (ya calculado en analyze_text())
            'Flesch Score': text_analysis['Flesch Score'],
            'Lexical Entropy': text_analysis['Lexical Entropy'],
            'Syntactic Repetition': text_analysis['Syntactic Repetition'],
            'Unusual Word Frequency': text_analysis['Unusual Word Frequency'],

            # Ironía y sátira (ya calculado en extract_features() y satira())
            'irony_score': basic_features['irony_score'],

        }
        return combined_features

    def predict_satire(self, text):
        selected_features = [
            'MeanWordLen',
            'LexicalDiversity',
            'MeanSentenceLen',
            'StdevSentenceLen',
            'DocumentLen',
            'WordsPerText',
            'SentencesPerText',
            'num_words',
            'num_chars',
            'irony_score',
            'prop_NOUN',
            'prop_VERB',
            'prop_ADJ',
            'rhetorical_questions',
            'avg_depth',
            'Flesch Score',
            'Lexical Entropy',
            'Syntactic Repetition',
            'Unusual Word Frequency',
        ]
        threshold = 0.55
        
        # Si estamos en modo de demostración (mock), usar predicción simulada para evitar errores
        if getattr(self.model_loader, 'is_mock', False):
            text_lower = text.lower()
            satire_indicators = ["obviamente", "claro", "supuestamente", "genial", "increíble", "absurdo", "ya que estamos", "de forma exagerada"]
            score = 0.2 + sum(0.15 for word in satire_indicators if word in text_lower)
            score = min(0.95, max(0.05, score))
            prediction = "ES SÁTIRA" if score >= threshold else "NO ES SÁTIRA"
            mock_metrics = {
                "irony_score": 0.65 if "obviamente" in text_lower or "claro" in text_lower else 0.15,
                "LexicalDiversity": 85.0,
                "Flesch Score": 75.0,
                "Unusual Word Frequency": 0.35,
                "prop_NOUN": 0.25,
                "prop_VERB": 0.18,
                "prop_ADJ": 0.12
            }
            return prediction, score, mock_metrics

        # 1. Preprocesar texto para características (TF-IDF y features manuales)
        processed_text = self.preprocess_text(text)

        with self.inference_lock:
            with torch.no_grad():
                # 2. Generar embeddings de BERT
                inputs = self.model_loader.tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
                outputs = self.model_loader.model.bert(**inputs)
                cls_output = outputs.last_hidden_state[:, 0, :]
                
                # 3. Obtener características de TF-IDF
                tfidf_features = self.model_loader.vectorizer.transform([processed_text]).toarray()
               
                # 4. Obtener características manuales
                features_dict = self.calculate_features(text)
                features_values = [features_dict[k] for k in selected_features]

                # 5. Combinar características (TF-IDF + manuales)
                combined_features_np = np.concatenate([tfidf_features, np.array([features_values])], axis=1)
                combined_features_normalized = self.model_loader.scaler.transform(combined_features_np)

                # 7. Convertir a tensor
                features_tensor = torch.tensor(combined_features_normalized, dtype=torch.float32).to(self.device)

                # 8. Concatenar embeddings BERT + características
                combined = torch.cat((cls_output, features_tensor), dim=1)

                # 9. Pasar por el modelo
                output = self.model_loader.model.softmax(self.model_loader.model.fc2(self.model_loader.model.relu(self.model_loader.model.fc1(combined))))
                prob_satira = torch.exp(output)[0][1].item()

        if prob_satira < threshold:
            prediction = "NO ES SÁTIRA"
        else:
            prediction = "ES SÁTIRA"

        return prediction, prob_satira, features_dict 

