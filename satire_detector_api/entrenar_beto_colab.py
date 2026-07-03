import os
import sys
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast

# 1. Configurar Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satire_detector_api.settings')

import django
django.setup()

from detector.utils.bert_classifier import BertClassifier

# 2. Cargar tokenizador, modelo, serializadores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print(f"DISPOSITIVO DE ENTRENAMIENTO COMPATIBLE: {device}")
print("="*60)

# Rutas locales
base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(os.path.dirname(base_path), "documentos_origen", "Titulacion1", "DatasetsFinales", "df_train2_featselect2.jsonl")
static_dir = os.path.join(base_path, "static")

tokenizer_path = os.path.join(static_dir, "model_files", "tokenizer_files")
model_path = os.path.join(static_dir, "best_model_spanish_loss.pt")
tfidf_path = os.path.join(static_dir, "tfidf_vectorizer.pkl")
scaler_path = os.path.join(static_dir, "minmax_scaler.pkl")

# Cargar dataset
print(f"[LOAD] Cargando dataset desde {dataset_path}...")
df = pd.read_json(dataset_path, orient='records', lines=True)
print(f"[INFO] Dataset cargado: {len(df)} registros.")

# Cargar Tokenizador
if os.path.exists(tokenizer_path):
    print(f"[LOAD] Cargando tokenizador local desde {tokenizer_path}...")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
else:
    print(f"[LOAD] Tokenizador local no encontrado. Descargando desde Hugging Face Hub (dccuchile/bert-base-spanish-wwm-uncased)...")
    tokenizer = BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

# Tokenizar transcripciones
print("[TOKENIZE] Tokenizando textos...")
encodings = tokenizer(
    df['transcription_processed'].tolist(),
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors="pt"
)

# Cargar serializadores
print(f"[LOAD] Cargando serializadores vectorizador y escalador...")
vectorizer = joblib.load(tfidf_path)
scaler = joblib.load(scaler_path)

# Extraer y normalizar características combinadas
print("[TRANSFORM] Calculando vectores combinados normalizados...")
tfidf_feats = vectorizer.transform(df['transcription_processed'].fillna("")).toarray()
manual_features = [
    'MeanWordLen', 'LexicalDiversity', 'MeanSentenceLen', 'StdevSentenceLen', 'DocumentLen',
    'WordsPerText', 'SentencesPerText', 'num_words', 'num_chars', 'irony_score',
    'prop_NOUN', 'prop_VERB', 'prop_ADJ', 'rhetorical_questions', 'avg_depth',
    'Flesch Score', 'Lexical Entropy', 'Syntactic Repetition', 'Unusual Word Frequency'
]
manual_feats = df[manual_features].values
combined = np.concatenate([tfidf_feats, manual_feats], axis=1)
combined_normalized = scaler.transform(combined)

input_ids = encodings['input_ids']
attention_masks = encodings['attention_mask']
extra_features = torch.tensor(combined_normalized, dtype=torch.float32)
labels = torch.tensor(df['label'].values, dtype=torch.long)

# 3. Partición de Datos (80/20 Stratified)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

indices = np.arange(len(labels))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=labels.numpy()
)

# Datasets & Dataloaders
train_dataset = TensorDataset(
    input_ids[train_idx],
    attention_masks[train_idx],
    extra_features[train_idx],
    labels[train_idx]
)
val_dataset = TensorDataset(
    input_ids[test_idx],
    attention_masks[test_idx],
    extra_features[test_idx],
    labels[test_idx]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"[INFO] Tamaño del subconjunto de Entrenamiento: {len(train_idx)} muestras")
print(f"[INFO] Tamaño del subconjunto de Validación (Test): {len(test_idx)} muestras")

# Cargar modelo previo o inicializar desde cero
import __main__
__main__.BertClassifier = BertClassifier

if os.path.exists(model_path):
    print(f"[LOAD] Cargando pesos previos del modelo desde {model_path}...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    learning_rate = 3e-6  # Tasa de aprendizaje baja para fine-tuning incremental
else:
    print(f"[INIT] Archivo de modelo previo no encontrado en {model_path}.")
    print(f"[INIT] Inicializando nuevo modelo híbrido BETO desde cero...")
    from transformers import AutoModel
    bert_base = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    model = BertClassifier(bert_base, num_extra_features=3019)
    learning_rate = 2e-5  # Tasa de aprendizaje estándar para entrenamiento completo

model.to(device)

# 4. Configurar optimizador y pérdida
epochs = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

print("\n" + "="*40)
print("INICIANDO RE-ENTRENAMIENTO (FINE-TUNING)")
print("="*40)

best_val_loss = float('inf')
best_model_state = None

for epoch in range(epochs):
    # Ciclo de entrenamiento
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        b_input_ids = batch[0].to(device)
        b_attn_mask = batch[1].to(device)
        b_extra = batch[2].to(device)
        b_labels = batch[3].to(device)
        
        model.zero_grad()
        outputs = model(b_input_ids, b_attn_mask, b_extra)
        loss = loss_fn(outputs, b_labels)
        total_train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Ciclo de validación (unseen test split)
    model.eval()
    total_val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            b_extra = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            outputs = model(b_input_ids, b_attn_mask, b_extra)
            loss = loss_fn(outputs, b_labels)
            total_val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(b_labels.cpu().numpy())
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    
    print(f"Época {epoch+1} / {epochs}:")
    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val Loss:   {avg_val_loss:.4f} | Val Accuracy: {val_accuracy * 100:.2f}%")
    
    # Guardar estado si es el mejor loss de validación
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Guardar en CPU para portabilidad en despliegue
        model.to("cpu")
        import copy
        best_model_state = copy.deepcopy(model)
        model.to(device)
        print("   -> ¡Nuevo mejor modelo guardado en memoria!")

# Guardar el mejor modelo en disco
print(f"\n[SAVE] Guardando el mejor modelo entrenado en {model_path}...")
torch.save(best_model_state, model_path)
print("="*60)
print("ENTRENAMIENTO COMPLETADO Y MODELO GUARDADO EXITOSAMENTE")
print("="*60)
