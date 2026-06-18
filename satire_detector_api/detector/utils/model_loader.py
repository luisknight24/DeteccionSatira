"""import os
import joblib
import torch
import pandas as pd
from transformers import BertTokenizerFast
from django.conf import settings
from transformers import AutoModel, AutoTokenizer

import torch.nn as nn
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_extra_features):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(bert_model.config.hidden_size + num_extra_features, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #cls_output = outputs.pooler_output
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = torch.cat((cls_output, extra_features), dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.softmax(self.fc2(x))

from detector.utils.model_loader import BertClassifier
class ModelLoader:


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        # Rutas base
        model_dir = os.path.join(settings.STATIC_ROOT , 'model_files')
        model_dir1 = os.path.join(settings.STATIC_ROOT )
        data_dir = os.path.join(settings.BASE_DIR, 'data')

        print(f"[DEBUG] Ruta modelo_dir: {model_dir}")
        print(f"[DEBUG] Ruta data_dir: {data_dir}")

        # Rutas individuales
        tokenizer_path = os.path.join(model_dir, 'tokenizer_files')
        vectorizer_path = os.path.join(model_dir1, 'tfidf_vectorizer.pkl')
        scaler_path = os.path.join(model_dir1, 'minmax_scaler.pkl')
        model_path = os.path.join(model_dir1, 'best_model_spanish_loss.pt')
        satirical_words_path = os.path.join(model_dir, 'adverbios_conectores_satira_expandido.csv')
        common_words_path = os.path.join(model_dir, 'CREA_PalabrasComunes.txt')
        historial_csv_path = os.path.join(data_dir, 'historial_entrenamientos.csv')

        print(f"[DEBUG] Ruta tokenizer: {tokenizer_path}")
        print(f"[DEBUG] Ruta vectorizer: {vectorizer_path}")
        print(f"[DEBUG] Ruta scaler: {scaler_path}")
        print(f"[DEBUG] Ruta modelo BERT: {model_path}")
        print(f"[DEBUG] Ruta palabras satíricas: {satirical_words_path}")
        print(f"[DEBUG] Ruta palabras comunes: {common_words_path}")
        print(f"[DEBUG] Ruta historial CSV: {historial_csv_path}")

        # Cargar tokenizer y modelos
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.scaler = joblib.load(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()

        # Cargar archivo de palabras satíricas
        satirical_df = pd.read_csv(satirical_words_path, sep=';')
        self.satirical_words = satirical_df['PALABRAS'].dropna().tolist()

        # Cargar archivo de palabras comunes
        df_CREA = pd.read_csv(common_words_path, sep='\t', encoding='latin-1')
        self.common_words_es = set(df_CREA['Palabra'].dropna().tolist())

        # Ruta CSV
        self.ruta_csv = historial_csv_path
"""
import os
import joblib
import torch
import pandas as pd
from transformers import BertTokenizerFast
from django.conf import settings

from .bert_classifier import BertClassifier  # Importar la clase BertClassifier

class ModelLoader:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        # Rutas base
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = os.path.join(settings.STATICFILES_DIRS[0] , 'model_files')
        model_dir1 = os.path.join(settings.STATICFILES_DIRS[0])
        data_dir = os.path.join(settings.BASE_DIR, 'data')

        #print(f"[DEBUG] Ruta modelo_dir: {model_dir}")
        #print(f"[DEBUG] Ruta data_dir: {data_dir}")

        # Rutas individuales
        tokenizer_path = os.path.join(model_dir, 'tokenizer_files')
        vectorizer_path = os.path.join(model_dir1, 'tfidf_vectorizer.pkl')
        scaler_path = os.path.join(model_dir1, 'minmax_scaler.pkl')
        model_path = os.path.join(model_dir1, 'best_model_spanish_loss.pt')
        satirical_words_path = os.path.join(model_dir1, 'adverbios_conectores_satira_expandido.csv')
        common_words_path = os.path.join(model_dir1, 'CREA_PalabrasComunes.txt')
        historial_csv_path = os.path.join(model_dir1, 'historial_entrenamientos.csv')
        
        #print(f"[DEBUG] Ruta tokenizer: {tokenizer_path}")
        #print(f"[DEBUG] Ruta vectorizer: {vectorizer_path}")
        #print(f"[DEBUG] Ruta scaler: {scaler_path}")
        #print(f"[DEBUG] Ruta modelo BERT: {model_path}")
        #print(f"[DEBUG] Ruta palabras satíricas: {satirical_words_path}")
        #print(f"[DEBUG] Ruta palabras comunes: {common_words_path}")
        #print(f"[DEBUG] Ruta historial CSV: {historial_csv_path}")

        # Cargar tokenizer y modelos
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.scaler = joblib.load(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("✅ Usando dispositivo:", self.device)
        from .bert_classifier import BertClassifier
        import __main__
        __main__.BertClassifier = BertClassifier
            # Ahora carga el modelo
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()


        # Cargar archivo de palabras satíricas
        satirical_df = pd.read_csv(satirical_words_path, sep=';')
        self.satirical_words = satirical_df['PALABRAS'].dropna().tolist()

        # Cargar archivo de palabras comunes
        df_CREA = pd.read_csv(common_words_path, sep='\t', encoding='latin-1')
        self.common_words_es = set(df_CREA['Palabra'].dropna().tolist())

        # Guardar ruta CSV
        self.ruta_csv = historial_csv_path
