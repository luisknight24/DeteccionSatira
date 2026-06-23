import os
import joblib
import torch
import pandas as pd
from transformers import BertTokenizerFast
from django.conf import settings
from .bert_classifier import BertClassifier

class MockTokenizer:
    def __call__(self, text, *args, **kwargs):
        return {
            "input_ids": torch.zeros((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long)
        }

class MockBERTModel(torch.nn.Module):
    class Config:
        hidden_size = 768
    def __init__(self):
        super().__init__()
        self.config = self.Config()
    def forward(self, *args, **kwargs):
        class Output:
            last_hidden_state = torch.zeros((1, 10, 768))
        return Output()

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = MockBERTModel()
        self.fc1 = torch.nn.Linear(768 + 3019, 256)
        self.fc2 = torch.nn.Linear(256, 2)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)
    def eval(self):
        pass

class MockVectorizer:
    def transform(self, texts):
        import scipy.sparse
        return scipy.sparse.csr_matrix((len(texts), 3000))

class MockScaler:
    def transform(self, X):
        return X

class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        self.is_mock = False
        
        # Base paths
        model_dir = os.path.join(settings.STATICFILES_DIRS[0], 'model_files')
        model_dir1 = os.path.join(settings.STATICFILES_DIRS[0])
        
        # Individual paths
        tokenizer_path = os.path.join(model_dir, 'tokenizer_files')
        vectorizer_path = os.path.join(model_dir1, 'tfidf_vectorizer.pkl')
        scaler_path = os.path.join(model_dir1, 'minmax_scaler.pkl')
        model_path = os.path.join(model_dir1, 'best_model_spanish_loss.pt')
        satirical_words_path = os.path.join(model_dir1, 'adverbios_conectores_satira_expandido.csv')
        common_words_path = os.path.join(model_dir1, 'CREA_PalabrasComunes.txt')
        historial_csv_path = os.path.join(model_dir1, 'historial_entrenamientos.csv')

        # Check if files exist, if not run in mock mode
        required_files = [vectorizer_path, scaler_path, model_path, satirical_words_path, common_words_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files or not os.path.exists(tokenizer_path):
            print("[ADVERTENCIA] Faltan archivos del modelo entrenado. Iniciando en MODO DEMOSTRACIÓN (MOCK).")
            print(f"Archivos faltantes: {missing_files if missing_files else ['tokenizer_files']}")
            self.is_mock = True
            self.tokenizer = MockTokenizer()
            self.vectorizer = MockVectorizer()
            self.scaler = MockScaler()
            self.model = MockModel()
            self.device = torch.device("cpu")
            
            # Load satire words fallback
            if os.path.exists(satirical_words_path):
                satirical_df = pd.read_csv(satirical_words_path, sep=';')
                self.satirical_words = satirical_df['PALABRAS'].dropna().tolist()
            else:
                self.satirical_words = ["obviamente", "claro", "supuestamente", "genial", "increíble", "absurdo"]

            # Load common words fallback
            if os.path.exists(common_words_path):
                df_CREA = pd.read_csv(common_words_path, sep='\t', encoding='latin-1')
                self.common_words_es = set(df_CREA['Palabra'].dropna().tolist())
            else:
                self.common_words_es = set()
                
            self.ruta_csv = historial_csv_path
            return

        # Regular loading path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[OK] Usando dispositivo:", self.device)
        
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.scaler = joblib.load(scaler_path)
            
            import __main__
            __main__.BertClassifier = BertClassifier
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.eval()

            # Load satirical words
            satirical_df = pd.read_csv(satirical_words_path, sep=';')
            self.satirical_words = satirical_df['PALABRAS'].dropna().tolist()

            # Load common words
            df_CREA = pd.read_csv(common_words_path, sep='\t', encoding='latin-1')
            self.common_words_es = set(df_CREA['Palabra'].dropna().tolist())

            self.ruta_csv = historial_csv_path
        except Exception as e:
            print(f"[ERROR] Error al cargar los modelos reales: {e}. Iniciando en modo fallback.")
            self.is_mock = True
            self.tokenizer = MockTokenizer()
            self.vectorizer = MockVectorizer()
            self.scaler = MockScaler()
            self.model = MockModel()
            self.device = torch.device("cpu")
            self.satirical_words = ["obviamente", "claro", "supuestamente", "genial", "increíble", "absurdo"]
            self.common_words_es = set()
            self.ruta_csv = historial_csv_path
