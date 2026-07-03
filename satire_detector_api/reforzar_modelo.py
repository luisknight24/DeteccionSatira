import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# 1. Configurar el entorno de Django para poder importar utilidades
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satire_detector_api.settings')

import django
django.setup()

from detector.utils.text_processor import TextProcessor

# 2. Definición del dataset de refuerzo (50 no-sátiras cotidianas/formales, 50 sátiras paródicas)
ejemplos_refuerzo = [
    # --- NO SÁTIRA (Factual, laboral, noticias, ciencia, RAE) ---
    {"text": "La última actualización del sistema operativo corrige doce vulnerabilidades de seguridad e incorpora mejoras en la gestión de la memoria RAM.", "label": 0},
    {"text": "La empresa ha organizado un programa de formación en habilidades digitales para sus empleados, con una duración total de cuarenta horas distribuidas en ocho semanas.", "label": 0},
    {"text": "Según un estudio reciente, establecer horarios fijos y reducir las interrupciones digitales puede aumentar la productividad laboral hasta un 23%.", "label": 0},
    {"text": "El salario medio en España se situó en 2.128 euros brutos mensuales en el último trimestre, lo que representa un incremento del 4,2% respecto al año anterior.", "label": 0},
    {"text": "La Real Academia Española presentó ayer las novedades de la vigesimocuarta edición de su diccionario, que incorpora 3.152 términos y modificaciones.", "label": 0},
    {"text": "Un estudiante de la Universidad Complutense defendió su tesis doctoral sobre comunicación digital, obteniendo la calificación de sobresaliente cum laude.", "label": 0},
    {"text": "El Ministerio de Sanidad ha iniciado una campaña de vacunación contra la gripe estacional dirigida a mayores de 65 años y personal sanitario.", "label": 0},
    {"text": "El comité de empresa y la dirección firmaron un acuerdo que contempla un aumento salarial del 3% y mejoras en las condiciones de teletrabajo.", "label": 0},
    {"text": "La conferencia sobre inteligencia artificial aplicada a la medicina reunirá a más de quinientos investigadores en el palacio de congresos mañana por la mañana.", "label": 0},
    {"text": "El telescopio espacial ha capturado nuevas imágenes de una galaxia espiral situada a más de cincuenta millones de años luz de la Tierra.", "label": 0},
    {"text": "La inflación interanual se moderó dos décimas en el mes de octubre, situándose en el 3,5% debido al abaratamiento de los carburantes.", "label": 0},
    {"text": "El ayuntamiento abrirá tres nuevas escuelas infantiles públicas el próximo curso, ofreciendo un total de doscientas cincuenta plazas gratuitas.", "label": 0},
    {"text": "La multinacional tecnológica invertirá sesenta millones de euros en su nuevo centro de investigación y desarrollo en Madrid.", "label": 0},
    {"text": "El Museo del Prado inaugura una exposición temporal con cuarenta obras maestras del Renacimiento italiano procedentes de colecciones privadas.", "label": 0},
    {"text": "Investigadores de la universidad desarrollan un nuevo material biodegradable a partir de residuos agrícolas que reduce el uso de plástico.", "label": 0},
    {"text": "La tasa de desempleo bajó al 11,6% en el tercer trimestre, registrando la cifra más baja en los últimos quince años según la encuesta de población activa.", "label": 0},
    {"text": "La Comisión Europea aprobó una nueva normativa de protección de datos para reforzar la privacidad de los usuarios en internet.", "label": 0},
    {"text": "El transporte público gratuito para jóvenes y desempleados entrará en vigor a partir del primero de enero en todo el territorio nacional.", "label": 0},
    {"text": "La cumbre del clima concluyó con un acuerdo firmado por ciento noventa países para acelerar la transición hacia energías renovables.", "label": 0},
    {"text": "El banco central decidió mantener los tipos de interés en el 4,5% para consolidar la estabilización de los precios de consumo.", "label": 0},
    {"text": "La farmacéutica inició la fase tres del ensayo clínico de su vacuna contra la malaria en siete países africanos simultáneamente.", "label": 0},
    {"text": "El sector turístico prevé una ocupación superior al 85% durante el puente de diciembre, superando las cifras del año pasado.", "label": 0},
    {"text": "La nueva ley de vivienda regula los precios de los alquileres en las zonas tensionadas para facilitar el acceso a la población joven.", "label": 0},
    {"text": "Científicos detectan un incremento del 1,5 por ciento en el nivel del mar en la costa norte debido al deshielo acelerado de los glaciares.", "label": 0},
    {"text": "La biblioteca municipal amplía su catálogo con más de dos mil títulos digitales y audiolibros disponibles para préstamo en línea.", "label": 0},
    {"text": "El gobierno regional aprueba una partida presupuestaria de diez millones de euros para la digitalización de pequeñas y medianas empresas.", "label": 0},
    {"text": "La marca automotriz presentó su nuevo modelo de coche eléctrico con una autonomía certificada de quinientos kilómetros por carga.", "label": 0},
    {"text": "El festival internacional de cine proyectará sesenta películas de treinta países en su sección oficial a partir del próximo viernes.", "label": 0},
    {"text": "La red de carreteras del Estado contará con doscientos nuevos puntos de recarga rápida para vehículos eléctricos en autovías principales.", "label": 0},
    {"text": "El observatorio meteorológico alerta de un descenso térmico de diez grados a partir del miércoles por la entrada de una masa de aire frío.", "label": 0},
    {"text": "La contaminación ambiental es uno de los principales problemas actuales. Reducir la cantidad de residuos y reciclar contribuye a proteger el medio ambiente.", "label": 0},
    {"text": "El transporte público permite movilizar a un gran número de personas de forma económica. Sin embargo, en las horas de mayor demanda suele presentarse congestión.", "label": 0},
    {"text": "El Congreso aprobó ayer, con 187 votos a favor y 52 en contra, la nueva ley de modificación del reglamento parlamentario.", "label": 0},
    {"text": "Una empresa tecnológica lanzó una nueva aplicación de productividad personal que permite sincronizar calendarios, listas de tareas y objetivos semanales en una sola interfaz.", "label": 0},
    {"text": "El nuevo asistente virtual incorpora funciones de inteligencia artificial para automatizar tareas administrativas, requiriendo únicamente permisos básicos de calendario y correo electrónico.", "label": 0},
    {"text": "Un informe financiero recomienda reducir los gastos mensuales no esenciales, como suscripciones digitales y ocio, para aumentar la tasa de ahorro personal hasta un 15% de los ingresos.", "label": 0},
    {"text": "Especialistas del sueño recomiendan evitar pantallas y establecer rutinas de desconexión digital al menos una hora antes de dormir para mejorar la calidad del descanso nocturno.", "label": 0},
    {"text": "El uso constante de dispositivos electrónicos puede generar fatiga visual y dolores musculares debido a malas posturas prolongadas.", "label": 0},
    {"text": "La transición hacia energías limpias como la solar y la eólica avanza a un ritmo constante en la región para reducir la huella de carbono.", "label": 0},
    {"text": "El teletrabajo ha transformado la dinámica laboral, permitiendo a los empleados conciliar su vida familiar y reducir los tiempos de desplazamiento diario.", "label": 0},
    {"text": "La nutrición equilibrada basada en alimentos frescos y la reducción de azúcares procesados es fundamental para prevenir enfermedades crónicas.", "label": 0},
    {"text": "La secretaría de tránsito implementará nuevos carriles exclusivos para bicicletas con el fin de promover la movilidad sostenible en el centro de la ciudad.", "label": 0},
    {"text": "Un estudio revela que los hábitos de lectura diarios estimulan la actividad cognitiva y mejoran la capacidad de concentración en jóvenes.", "label": 0},
    {"text": "El Banco Central anunció un programa de educación financiera gratuito para enseñar a las familias a crear presupuestos y gestionar sus deudas.", "label": 0},
    {"text": "La cumbre de innovación educativa reunirá a profesores de todo el continente para debatir metodologías de enseñanza activa en las aulas.", "label": 0},
    {"text": "La instalación de filtros de aire en las escuelas ayuda a reducir la concentración de partículas contaminantes y mejora el rendimiento de los alumnos.", "label": 0},
    {"text": "La OMS recuerda la importancia de realizar al menos 150 minutos de actividad física moderada a la semana para mantener una buena salud cardiovascular.", "label": 0},
    {"text": "El sector agrícola busca alternativas sostenibles frente a las sequías mediante sistemas de riego por goteo automatizados de alta precisión.", "label": 0},
    {"text": "Un nuevo acuerdo comercial entre ambos países eliminará aranceles para productos tecnológicos y agrícolas a partir del próximo mes.", "label": 0},
    {"text": "La universidad pública abre el proceso de admisión para el próximo semestre académico con una oferta de treinta y dos carreras de pregrado.", "label": 0},

    # --- SÁTIRA (Parodias de noticias, humor de redes sociales, ironía laboral) ---
    {"text": "La nueva actualización del sistema operativo de Microsoft corrige doce vulnerabilidades e incorpora cincuenta nuevos bugs para mantener entretenidos a los usuarios.", "label": 1},
    {"text": "La empresa organiza un curso obligatorio de cuarenta horas sobre cómo parecer ocupado en Teams mientras duermes la siesta.", "label": 1},
    {"text": "Según un estudio reciente, mirar fijamente a la pantalla del ordenador con cara de frustración aumenta tu salario un 23% sin trabajar más.", "label": 1},
    {"text": "El salario medio en mi cuenta corriente se situó en 2 euros mensuales, un incremento del 0,1% que me permitirá comprar medio chicle este año.", "label": 1},
    {"text": "La Real Academia Española presenta su nuevo diccionario con 3.000 palabras inventadas en Twitter para no quedarse atrás en la evolución del idioma.", "label": 1},
    {"text": "Un estudiante de la Universidad Complutense defiende su tesis doctoral escrita completamente con emojis, obteniendo la calificación de sobresaliente por parte del jurado moderno.", "label": 1},
    {"text": "El Ministerio de Sanidad inicia una campaña de vacunación masiva para inmunizar a la población contra las opiniones de sus cuñados en las cenas familiares.", "label": 1},
    {"text": "El departamento de recursos humanos aprueba un aumento salarial del 0% y el teletrabajo obligatorio los domingos de dos a cuatro de la madrugada.", "label": 1},
    {"text": "Científicos descubren que la cafeína es capaz de hacer que los lunes parezcan martes, aunque no cura la profunda tristeza de volver a la oficina.", "label": 1},
    {"text": "Un nuevo telescopio revela que los extraterrestres han decidido evitar la Tierra tras observar cómo nos comportamos en las rebajas de enero.", "label": 1},
    {"text": "El ayuntamiento inaugura tres nuevas guarderías para adultos donde puedes llorar por tu hipoteca y colorear dibujos sin que nadie te juzgue.", "label": 1},
    {"text": "La multinacional tecnológica presenta su nuevo dispositivo que emite descargas eléctricas leves cada vez que abres una pestaña de redes sociales en horas de trabajo.", "label": 1},
    {"text": "El Museo del Prado expone una colección de capturas de pantalla de chats de trabajo titulada 'El horror cotidiano del siglo veintiuno'.", "label": 1},
    {"text": "Científicos desarrollan un café biodegradable que se bebe solo cuando detecta que tu jefe se acerca a tu mesa a pedirte un informe urgente.", "label": 1},
    {"text": "La tasa de optimismo nacional cae al 0,5% tras enterarnos de que el precio de los aguacates ha vuelto a subir tres céntimos.", "label": 1},
    {"text": "La Comisión Europea aprueba una ley que prohíbe los correos electrónicos que empiecen con 'espero que estés bien' para proteger la salud mental colectiva.", "label": 1},
    {"text": "El transporte público será gratuito para personas que prometan no escuchar música en altavoz ni hablar de su vida amorosa en el trayecto.", "label": 1},
    {"text": "La cumbre del clima concluye con el firme compromiso de seguir reuniéndose en resorts de lujo de todo el mundo para debatir por qué hace calor.", "label": 1},
    {"text": "El banco central mantiene los tipos de interés altos porque el director asegura que ahorrar nos hace débiles y que debemos consumir desesperadamente.", "label": 1},
    {"text": "La farmacéutica patenta una pastilla para tolerar las reuniones de dos horas que pudieron haberse solucionado con un simple correo electrónico.", "label": 1},
    {"text": "El sector turístico prevé una ocupación del 150% en las playas, obligando a los turistas a turnarse para poder poner un pie sobre la arena.", "label": 1},
    {"text": "La nueva ley de vivienda declara las cajas de cartón grandes como apartamentos tipo estudio aptos para jóvenes profesionales por novecientos euros al mes.", "label": 1},
    {"text": "Científicos confirman que el nivel de drama en las redes sociales subió un 40% después de que cayera el servidor de mensajería durante diez minutos.", "label": 1},
    {"text": "La biblioteca municipal incorpora un catálogo especial de libros que compraste, juraste leer y que ahora solo sirven para decorar tu estantería.", "label": 1},
    {"text": "El gobierno regional subvenciona con dos millones de euros un estudio para averiguar por qué los gatos nos ignoran de forma sistemática.", "label": 1},
    {"text": "La marca automotriz presenta su nuevo coche eléctrico que solo avanza si le hablas con cariño y le pides disculpas por haberlo dejado en la lluvia.", "label": 1},
    {"text": "El festival de cine otorgará el premio a la película más corta de la historia: una videollamada de trabajo donde todos tienen la cámara apagada.", "label": 1},
    {"text": "La red de carreteras instalará doscientos puntos de recarga de café rápido para conductores que viajan con sus familias durante las vacaciones.", "label": 1},
    {"text": "El observatorio meteorológico alerta de que el clima de mañana dependerá de la cantidad de personas que salgan hoy a la calle con paraguas.", "label": 1},
    {"text": "La última encuesta revela que el 99% de las personas que dicen 'llego en cinco minutos' aún están en su cama eligiendo qué ponerse.", "label": 1},
    {"text": "El gobierno anuncia que la mejor forma de combatir la contaminación por plásticos es cobrar el doble por las bolsas y esperar que los peces aprendan a digerirlas.", "label": 1},
    {"text": "El transporte público de la ciudad implementará un nuevo sistema donde los pasajeros viajan colgados de las ventanas para ahorrar espacio y mejorar el flujo.", "label": 1},
    {"text": "El Congreso aprueba un aumento histórico del 200% en sus propios salarios tras debatir intensamente durante cinco minutos si el café de la cafetería estaba frío.", "label": 1},
    {"text": "Lanzan una nueva app de productividad que te envía una notificación de burla cada cinco minutos si detecta que estás mirando la pared en vez de trabajar.", "label": 1},
    {"text": "El nuevo robot con inteligencia artificial promete redactar tus correos de disculpa corporativa con un tono 100% de arrepentimiento falso garantizado.", "label": 1},
    {"text": "Un manual financiero te aconseja dejar de comer pan y suspender tu suscripción de música para que en 80 años puedas comprar el 1% de un apartamento.", "label": 1},
    {"text": "Científicos del sueño descubren que mirar el móvil con el brillo al máximo a las tres de la madrugada es una excelente forma de reflexionar sobre todos tus errores del pasado.", "label": 1},
    {"text": "Un estudio médico confirma que pasar diez horas al día encorvado frente a la pantalla te convertirá eventualmente en un majestuoso dinosaurio cibernético.", "label": 1},
    {"text": "La empresa petrolera asegura que su nuevo combustible es ecológico porque ahora las latas de los barriles son de color verde brillante.", "label": 1},
    {"text": "El teletrabajo es fantástico porque te permite sentir el estrés de la oficina y la soledad de tu hogar al mismo tiempo y sin costo adicional.", "label": 1},
    {"text": "Una nueva dieta revolucionaria consiste en comer únicamente aire purificado de los Pirineos para purificar el alma y vaciar la billetera.", "label": 1},
    {"text": "El ayuntamiento instalará ciclovías verticales en las paredes de los edificios para no molestar a los conductores de camionetas gigantes.", "label": 1},
    {"text": "Un estudio sociológico revela que las personas que leen libros en el metro lo hacen principalmente para evitar el contacto visual con otros humanos.", "label": 1},
    {"text": "El banco central te ofrece un curso de ahorro donde te enseñan a vivir bajo un puente para recortar los molestos gastos de alquiler.", "label": 1},
    {"text": "La nueva metodología educativa consiste en que los alumnos enseñen a los profesores a usar TikTok a cambio de una nota aprobatoria.", "label": 1},
    {"text": "Las escuelas instalarán plantas purificadoras que funcionan con la energía de los suspiros de aburrimiento de los estudiantes en clase de matemáticas.", "label": 1},
    {"text": "La OMS sugiere correr en círculos gritando de pánico durante 15 minutos al día como sustituto ideal de la actividad física moderada.", "label": 1},
    {"text": "Agricultores descubren que hablarle de forma pasivo-agresiva a las plantas de tomate aumenta su producción de forma notable bajo condiciones de sequía.", "label": 1},
    {"text": "El nuevo tratado comercial permitirá importar aire embotellado de primera calidad para sustituir la atmósfera local durante los días de tráfico intenso.", "label": 1},
    {"text": "La universidad ofrece un curso intensivo de cuatro años sobre cómo quejarse de la universidad en redes sociales con ortografía impecable.", "label": 1}
]

# Alineadas perfectamente con text_processor.py (mismo orden y nombres de clave)
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
    'Unusual Word Frequency'
]

def agregar_y_procesar():
    ruta_original = os.path.join("documentos_origen", "Titulacion1", "DatasetsFinales", "df_train2_featselect2.jsonl")
    
    if not os.path.exists(ruta_original):
        print(f"[ERROR] No se encontro el dataset original en {ruta_original}")
        return
        
    print(f"[LOAD] Cargando dataset original ({ruta_original})...")
    df_original = pd.read_json(ruta_original, orient='records', lines=True)
    print(f"[INFO] Dataset cargado. Contiene {len(df_original)} registros.")

    processor = TextProcessor()
    nuevos_registros = []
    
    print("\n[INFO] Procesando caracteristicas linguisticas de los nuevos ejemplos de refuerzo...")
    for idx, ejemplo in enumerate(ejemplos_refuerzo):
        text = ejemplo["text"]
        label = ejemplo["label"]
        
        # Generar texto procesado (lematización + limpieza)
        processed_text = processor.preprocess_text(text)
        # Extraer las 19 características manuales
        features_dict = processor.calculate_features(text)
        
        # Armar el registro equivalente a las columnas del JSONL
        registro = {
            "id": f"refuerzo_{idx}",
            "transcription": text,
            "transcription_processed": processed_text,
            "label": label
        }
        
        # Añadir las características calculadas al registro en el orden correcto
        for feat in selected_features:
            registro[feat] = features_dict.get(feat, 0)
            
        nuevos_registros.append(registro)
        
    df_nuevos = pd.DataFrame(nuevos_registros)
    print(f"[INFO] Procesados {len(df_nuevos)} nuevos registros de refuerzo.")
    
    # Combinar datasets
    df_expandido = pd.concat([df_original, df_nuevos], ignore_index=True)
    
    # Guardar el dataset expandido
    df_expandido.to_json(ruta_original, orient='records', lines=True)
    print(f"[SAVE] Dataset expandido guardado en {ruta_original} (Total: {len(df_expandido)} registros).")
    
    # 3. Re-entrenar y guardar los serializadores locales (TF-IDF y MinMaxScaler)
    print("\n[INFO] Re-ajustando Vectorizador TF-IDF y Escalador MinMaxScaler con el corpus expandido...")
    
    # TF-IDF con los mismos hiperparámetros que el original (ngram_range=(1, 1))
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_features = vectorizer.fit_transform(df_expandido['transcription_processed'].fillna("")).toarray()
    
    # Extraer las 19 características y concatenar con TF-IDF
    manual_features = df_expandido[selected_features].values
    combined_features = np.concatenate([tfidf_features, manual_features], axis=1)
    
    # Escalar
    scaler = MinMaxScaler()
    scaler.fit(combined_features)
    
    # Guardar localmente para subirlos
    static_dir = os.path.join("satire_detector_api", "static")
    os.makedirs(static_dir, exist_ok=True)
    
    joblib.dump(vectorizer, os.path.join(static_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(static_dir, "minmax_scaler.pkl"))
    print(f"[INFO] tfidf_vectorizer.pkl y minmax_scaler.pkl actualizados con exito en {static_dir}!")
    
    print("\n" + "="*60)
    print("INSTRUCCIONES DE RE-ENTRENAMIENTO (GOOGLE COLAB)")
    print("="*60)
    print("Para entrenar el modelo BETO con tus nuevos datos, sigue estos pasos en Colab:")
    print("1. Ejecuta Git Pull en Colab para bajar el dataset expandido.")
    print("2. Abre tu celda de entrenamiento de BETO (o crea una nueva) y ejecuta este bloque:")
    print("-" * 60)
    print("""
# --- CODIGO DE COLAB PARA AJUSTE FINO INCREMENTAL ---
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import joblib
from transformers import BertTokenizerFast, AutoModel

# 1. Cargar el dataset con las nuevas muestras agregadas
df = pd.read_json("/content/DeteccionSatira/documentos_origen/Titulacion1/DatasetsFinales/df_train2_featselect2.jsonl", orient='records', lines=True)

# 2. Cargar tokenizador de BETO
tokenizer = BertTokenizerFast.from_pretrained("/content/DeteccionSatira/satire_detector_api/static/model_files/tokenizer_files")
encodings = tokenizer(df['transcription_processed'].tolist(), truncation=True, padding=True, max_length=64, return_tensors="pt")

# 3. Preparar caracteristicas combinadas usando los nuevos pkl
vectorizer = joblib.load("/content/DeteccionSatira/satire_detector_api/static/tfidf_vectorizer.pkl")
scaler = joblib.load("/content/DeteccionSatira/satire_detector_api/static/minmax_scaler.pkl")

tfidf_feats = vectorizer.transform(df['transcription_processed'].fillna("")).toarray()
manual_feats = df[[
    'MeanWordLen', 'LexicalDiversity', 'MeanSentenceLen', 'StdevSentenceLen', 'DocumentLen',
    'WordsPerText', 'SentencesPerText', 'num_words', 'num_chars', 'irony_score',
    'prop_NOUN', 'prop_VERB', 'prop_ADJ', 'rhetorical_questions', 'avg_depth',
    'Flesch Score', 'Lexical Entropy', 'Syntactic Repetition', 'Unusual Word Frequency'
]].values

combined = np.concatenate([tfidf_feats, manual_feats], axis=1)
combined_normalized = scaler.transform(combined)

input_ids = encodings['input_ids']
attention_masks = encodings['attention_mask']
extra_features = torch.tensor(combined_normalized, dtype=torch.float32)
labels = torch.tensor(df['label'].values, dtype=torch.long)

# 4. Crear DataLoader
dataset = TensorDataset(input_ids, attention_masks, extra_features, labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 5. Cargar tu modelo actual para continuar el entrenamiento (Incremental Fine-tuning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Entrenando en:", device)

import sys
sys.path.append("/content/DeteccionSatira/satire_detector_api")
from detector.utils.bert_classifier import BertClassifier

# Cargar el modelo preexistente
model = torch.load("/content/DeteccionSatira/satire_detector_api/static/best_model_spanish_loss.pt", map_location=device, weights_only=False)
model.train()

# 6. Bucle de ajuste fino corto (3 epocas con Learning Rate bajo)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
loss_fn = nn.NLLLoss()

print("Iniciando ajuste fino de adaptacion...")
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        b_input_ids, b_input_mask, b_extra, b_labels = [t.to(device) for t in batch]
        
        model.zero_grad()
        outputs = model(b_input_ids, b_input_mask, b_extra)
        loss = loss_fn(outputs, b_labels)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    print(f"Epoca {epoch+1} - Perdida Promedio: {total_loss/len(train_loader):.4f}")

# 7. Guardar el modelo con los pesos adaptados
torch.save(model, "/content/DeteccionSatira/satire_detector_api/static/best_model_spanish_loss.pt")
print("Ajuste fino completado y guardado con exito!")
""")
    print("="*60)

if __name__ == "__main__":
    agregar_y_procesar()
