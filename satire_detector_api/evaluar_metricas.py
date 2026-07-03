import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# 1. Configurar entorno de Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satire_detector_api.settings')

import django
django.setup()

from detector.utils.text_processor import TextProcessor

selected_features = [
    'MeanWordLen', 'LexicalDiversity', 'MeanSentenceLen', 'StdevSentenceLen',
    'DocumentLen', 'WordsPerText', 'SentencesPerText', 'num_words', 'num_chars',
    'irony_score', 'prop_NOUN', 'prop_VERB', 'prop_ADJ', 'rhetorical_questions',
    'avg_depth', 'Flesch Score', 'Lexical Entropy', 'Syntactic Repetition', 'Unusual Word Frequency'
]

def evaluar():
    ruta_dataset = os.path.join("documentos_origen", "Titulacion1", "DatasetsFinales", "df_train2_featselect2.jsonl")
    if not os.path.exists(ruta_dataset):
        print(f"[ERROR] No se encontro el dataset en {ruta_dataset}")
        return

    print(f"[LOAD] Cargando dataset expandido de {ruta_dataset}...")
    df = pd.read_json(ruta_dataset, orient='records', lines=True)
    print(f"[INFO] Registros totales cargados: {len(df)}")
    print(f"[INFO] Distribucion de clases:\n{df['label'].value_counts()}")

    # 2. Ajustar TF-IDF
    print("\n[TRANSFORM] Calculando TF-IDF (3000 caracteristicas)...")
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_feats = vectorizer.fit_transform(df['transcription_processed'].fillna("")).toarray()

    # 3. Extraer e imputar caracteristicas manuales
    print("[TRANSFORM] Extrayendo caracteristicas manuales e imputando...")
    manual_feats = df[selected_features].values
    imputer = SimpleImputer(strategy='mean')
    manual_feats = imputer.fit_transform(manual_feats)

    # 4. Concatenar y Normalizar
    X_combined = np.concatenate([tfidf_feats, manual_feats], axis=1)
    y = df['label'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # 5. Particion Train/Test (80/20) con estratificacion
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Tamaño entrenamiento: {X_train.shape[0]}, Tamaño test: {X_test.shape[0]}")

    # Balancear con SMOTE
    print("[SMOTE] Aplicando SMOTE para balancear clases en entrenamiento...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[INFO] Clases despues de SMOTE: Sátiras={sum(y_train_res == 1)}, No-sátiras={sum(y_train_res == 0)}")

    # 6. Entrenar y evaluar modelos
    print("\n[TRAIN] Entrenando clasificadores tradicionales...")
    
    clf_svc = SVC(probability=True, kernel='linear', random_state=42)
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf_mlp = MLPClassifier(max_iter=500, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('svc', clf_svc), ('rf', clf_rf), ('xgb', clf_xgb), ('mlp', clf_mlp)],
        voting='soft'
    )

    modelos = {
        "Random Forest": clf_rf,
        "SVM Lineal": clf_svc,
        "XGBoost": clf_xgb,
        "Multilayer Perceptron (MLP)": clf_mlp,
        "Voting Classifier (Ensamble)": voting_clf
    }

    reportes = {}

    for nombre, modelo in modelos.items():
        print(f"-> Entrenando {nombre}...")
        modelo.fit(X_train_res, y_train_res)
        y_pred = modelo.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        
        reportes[nombre] = {
            "Accuracy": acc,
            "F1-Score": f1,
            "Precision": prec,
            "Recall": rec,
            "Full Report": classification_report(y_test, y_pred, digits=4)
        }
        print(f"   [OK] {nombre} finalizado (Accuracy: {acc:.4f}, F1: {f1:.4f})")

    # 7. Imprimir comparacion
    print("\n" + "="*70)
    print("REPORTE COMPARATIVO DE MODELOS (DATASET EXPANDIDO 6,100 REGISTROS)")
    print("="*70)
    for nombre, met in reportes.items():
        print(f"\nModelo: {nombre}")
        print(f"Accuracy:  {met['Accuracy'] * 100:.2f}%")
        print(f"F1-Score:  {met['F1-Score'] * 100:.2f}%")
        print(f"Precision: {met['Precision'] * 100:.2f}%")
        print(f"Recall:    {met['Recall'] * 100:.2f}%")
        print("-" * 50)
        print("Detalle de clasificacion:")
        print(met["Full Report"])
        print("="*70)

if __name__ == "__main__":
    evaluar()
