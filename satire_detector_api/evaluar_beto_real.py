import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# 1. Configurar entorno Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satire_detector_api.settings')

import django
django.setup()

from detector.utils.bert_classifier import BertClassifier

def evaluar_beto_real():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print(f"EVALUACIÓN REAL DEL MODELO EN DISPOSITIVO: {device}")
    print("="*60)

    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(base_path), "documentos_origen", "Titulacion1", "DatasetsFinales", "df_train2_featselect2.jsonl")
    static_dir = os.path.join(base_path, "static")

    tokenizer_path = os.path.join(static_dir, "model_files", "tokenizer_files")
    model_path = os.path.join(static_dir, "best_model_spanish_loss.pt")
    tfidf_path = os.path.join(static_dir, "tfidf_vectorizer.pkl")
    scaler_path = os.path.join(static_dir, "minmax_scaler.pkl")

    if not os.path.exists(model_path):
        print(f"[ERROR] No se encontró el archivo del modelo en {model_path}")
        return

    # Cargar dataset
    print(f"[LOAD] Cargando dataset desde {dataset_path}...")
    df = pd.read_json(dataset_path, orient='records', lines=True)
    
    # Cargar Tokenizador
    if os.path.exists(tokenizer_path):
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

    # Cargar serializadores
    vectorizer = joblib.load(tfidf_path)
    scaler = joblib.load(scaler_path)

    # 2. Partición de test 80/20 exacta (Stratified)
    selected_features = [
        'MeanWordLen', 'LexicalDiversity', 'MeanSentenceLen', 'StdevSentenceLen', 'DocumentLen',
        'WordsPerText', 'SentencesPerText', 'num_words', 'num_chars', 'irony_score',
        'prop_NOUN', 'prop_VERB', 'prop_ADJ', 'rhetorical_questions', 'avg_depth',
        'Flesch Score', 'Lexical Entropy', 'Syntactic Repetition', 'Unusual Word Frequency'
    ]

    X_text = df['transcription_processed'].fillna("").values
    X_manual = df[selected_features].values
    y = df['label'].values

    indices = np.arange(len(df))
    _, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )

    X_text_test = X_text[test_idx]
    X_manual_test = X_manual[test_idx]
    y_test = y[test_idx]

    # Preprocesar
    tfidf_feats = vectorizer.transform(X_text_test).toarray()
    combined = np.concatenate([tfidf_feats, X_manual_test], axis=1)
    combined_normalized = scaler.transform(combined)

    # Cargar modelo PyTorch
    import __main__
    __main__.BertClassifier = BertClassifier
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Inferencia
    print(f"[INFERENCE] Evaluando {len(y_test)} muestras de test...")
    all_preds = []
    batch_size = 32

    for i in range(0, len(X_text_test), batch_size):
        batch_texts = X_text_test[i:i+batch_size]
        batch_extra = combined_normalized[i:i+batch_size]

        encodings = tokenizer(
            batch_texts.tolist(),
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)

        extra_tensor = torch.tensor(batch_extra, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(encodings['input_ids'], encodings['attention_mask'], extra_tensor)
            probs = torch.exp(outputs)
            # Umbral de clasificación estándar a 0.55
            preds = (probs[:, 1] >= 0.55).cpu().numpy().astype(int)
            all_preds.extend(preds)

    all_preds = np.array(all_preds)

    # 3. Calcular Métricas
    acc = accuracy_score(y_test, all_preds)
    f1 = f1_score(y_test, all_preds, average='binary')
    prec = precision_score(y_test, all_preds, average='binary')
    rec = recall_score(y_test, all_preds, average='binary')
    cm = confusion_matrix(y_test, all_preds)

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*60)
    print("MÉTRICAS REALES OBTENIDAS DEL ENTRENAMIENTO EN COLAB")
    print("="*60)
    print(f"Exactitud Global (Accuracy):    {acc * 100:.2f}%")
    print(f"Medida F1-Score (Sátira):        {f1 * 100:.2f}%")
    print(f"Precisión (Clase Sátira):        {prec * 100:.2f}%")
    print(f"Sensibilidad (Recall):           {rec * 100:.2f}%")
    print("-"*60)
    print("MATRIZ DE CONFUSIÓN:")
    print(f"  Verdaderos Negativos (TN): {tn} (Textos neutros correctos)")
    print(f"  Falsos Positivos (FP):     {fp} (Textos serios erróneos)")
    print(f"  Falsos Negativos (FN):     {fn} (Sátiras no detectadas)")
    print(f"  Verdaderos Positivos (TP): {tp} (Sátiras correctas)")
    print("="*60)

if __name__ == "__main__":
    evaluar_beto_real()
