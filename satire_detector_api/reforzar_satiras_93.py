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

# 93 Ejemplos de sátira cotidiana / tweets / parodia laboral
ejemplos_satiricos_93 = [
    "Adoro que mi alarma decida apagarse sola justo el día del examen final, es muy aventurero.",
    "Qué divertido es que la cafetera se rompa el lunes a primera hora, ideal para entrenar la paciencia.",
    "Mi jefe quiere que hagamos sinergia en equipo, que en su idioma significa hacer horas extras sin cobrar.",
    "Qué gran invento el teletrabajo: ahora puedo estresarme en la comodidad de mis pijamas.",
    "Mi planta prefiere morir de sed antes que beber agua del grifo, muy refinada.",
    "El metro venía tan vacío hoy que solo tuve que compartir mi asiento con tres personas y una maleta.",
    "Ahorrar diez euros al mes me asegura comprar un café en la calle para cuando cumpla noventa años.",
    "Qué suerte tener un coche que hace ruidos extraños, es como tener una orquesta mecánica privada.",
    "Mi gato camina sobre el teclado para ayudarme a escribir correos indescifrables a la dirección.",
    "Adoro cuando los mensajeros juegan al escondite con mis paquetes y los dejan en otra calle.",
    "Qué oportuno que el internet se caiga justo cuando estoy en medio de una partida importante.",
    "Mi nevera tiene tanta luz y tan poca comida que parece un escaparate de minimalismo sueco.",
    "Qué alegría pagar impuestos para que reparen la calle principal por cuarta vez este mes.",
    "El gimnasio me cobra la membresía mensual para recordarme lo constante que soy en no ir.",
    "Mi perro ladra al aire a las tres de la mañana para asegurarse de que no haya fantasmas aburridos.",
    "Adoro las llamadas de telemarketing los sábados a las ocho de la mañana, muy considerados.",
    "Qué gran idea poner el botón de cerrar sesión oculto bajo cinco menús de configuración.",
    "Mi plan de jubilación consiste en esperar que los precios de las casas bajen un noventa por ciento.",
    "Qué suerte vivir en un estudio donde puedo tocar la nevera y la cama al mismo tiempo.",
    "Adoro cuando el médico me dice que no me estrese mientras me entrega una factura enorme.",
    "Mi planta de interior ha decidido marchitarse porque le hablé con un tono demasiado fuerte.",
    "Qué bien pensados los empaques de plástico que requieren tijeras especiales para ser abiertos.",
    "Mi gato me mira con cara de juzgarme por estar todo el día frente al computador sin cazar ratones.",
    "Adoro cuando la aplicación me pide actualizarse justo cuando tengo que mostrar el billete de tren.",
    "Qué oportuno que empiece a llover justo cuando decido salir a caminar sin paraguas.",
    "Mi plan para comer sano consiste en comprar verduras que luego dejaré pudrirse en la nevera con culpa.",
    "Qué gran invento las contraseñas que expiran cada quince días para mantenernos con la memoria activa.",
    "Adoro los correos que empiezan con 'revisar con urgencia' enviados el viernes a las cinco de la tarde.",
    "Mi coche eléctrico tiene una autonomía excelente, sobre todo cuando lo empujo cuesta abajo.",
    "Qué suerte tener vecinos que tocan la batería los domingos por la mañana, música clásica gratis.",
    "Adoro cuando la página web del gobierno me dice que mi trámite ha sido rechazado sin explicar la razón.",
    "Mi planta de menta ha decidido invadir la casa del vecino para buscar mejores condiciones de vida.",
    "Qué bien se siente pagar por una suscripción de streaming que solo tiene películas que ya vi.",
    "Mi perro considera que el mejor lugar para morder su hueso ruidoso es mi pie durante la reunión.",
    "Adoro cuando me dicen que sea creativo pero me dan una plantilla rígida de hace veinte años.",
    "Qué oportuno que el ascensor se descomponga el día que traigo las bolsas pesadas del supermercado.",
    "Mi nevera tiene tantas botellas de aderezos vacías que ya califica como museo de salsas históricas.",
    "Qué gran idea poner luces led de colores brillantes en un humidificador pensado para dormir.",
    "Adoro cuando me piden mi opinión en el trabajo para luego hacer exactamente lo contrario de lo sugerido.",
    "Mi planta de cactus ha decidido morir de exceso de sequía, un récord histórico de la botánica.",
    "Qué suerte tener un trabajo dinámico que me permite desarrollar la habilidad de no parpadear en tres horas.",
    "Adoro cuando la aerolínea pierde mi equipaje porque así tengo la oportunidad de renovar mi vestuario.",
    "Mi gato prefiere dormir sobre la caja de cartón de la cama cara que le compré por internet.",
    "Qué oportuno que el teléfono se quede sin batería justo cuando tengo que pagar con código QR.",
    "Mi plan de ahorro consiste en no mirar mi saldo bancario para no llenarme de preocupaciones inútiles.",
    "Adoro cuando los políticos prometen bajar los precios mientras suben sus propios sueldos un veinte por ciento.",
    "Qué bien pensada la burocracia: para cancelar un servicio tienes que enviar una carta escrita a mano.",
    "Mi perro ladra al cartero como si estuviera defendiendo la casa de una invasión alienígena inminente.",
    "Adoro cuando la comida saludable cuesta el triple que una pizza familiar, muy motivador.",
    "Qué gran invento el corrector ortográfico que cambia mis palabras correctas por términos absurdos.",
    "Mi planta de romero se ha secado porque no le gustó el ángulo en el que le daba el sol.",
    "Adoro cuando el seguro médico no cubre la única enfermedad que he tenido en toda mi vida.",
    "Qué oportuno que el vecino decida podar el césped a las seis de la mañana del sábado.",
    "Mi nevera está tan vacía que el hielo del congelador ya empezó a tomar un sabor a cebolla vieja.",
    "Qué bien se viaja en el metro en hora pico, un baño sauna interactivo con extraños sudorosos.",
    "Mi planta de aloe vera ha decidido pudrirse por exceso de atención, qué sensible de su parte.",
    "Adoro cuando el entrevistador me pregunta dónde me veo en cinco años y no puedo decir 'durmiendo'.",
    "Qué gran avance de las redes sociales: ahora puedo ver la vida perfecta de mis excompañeros mientras como fideos.",
    "Mi gato corre a máxima velocidad a las dos de la madrugada para recordarme que la noche es joven.",
    "Qué oportuno que el calentador falle justo cuando tengo champú en los ojos en pleno invierno.",
    "Adoro cuando me piden que trabaje horas extras el día de mi cumpleaños para demostrar mi compromiso.",
    "Mi perro persigue el reflejo de mi móvil en la pared con una constancia científica admirable.",
    "Qué suerte tener un trabajo donde puedo gestionar la frustración en tiempo real y gratis.",
    "Adoro cuando el botón de cancelar suscripción está escondido detrás de tres encuestas obligatorias.",
    "Mi cuenta corriente está en un estado de minimalismo tan profundo que no tiene dígitos de sobra.",
    "Qué bien pensados los empaques de galletas que se rompen por completo al intentar abrirlos.",
    "Mi planta se marchitó porque olvidé abrir la persiana durante una mañana de lluvia ligera.",
    "Adoro cuando el mensajero deja el paquete en el tejado y me envía una foto como prueba de entrega.",
    "Qué gran avance la domótica: ahora paso media hora intentando encender la bombilla con la aplicación.",
    "Mi perro muerde una botella plástica ruidosa justo cuando estoy en videollamada con la gerencia.",
    "Qué oportuno que cancelen mi vuelo justo el día de la entrevista para el trabajo ideal.",
    "Adoro cuando me piden autonomía pero tengo que pedir permiso para comprar un bolígrafo nuevo.",
    "Mi gato maúlla a la puerta cerrada del pasillo como si conversara con seres del más allá.",
    "Qué bien pensado el sistema de estacionamiento: pagas una fortuna por dejar el coche sobre baches.",
    "Adoro cuando me dan consejos de alimentación saludable personas con cocinero y nutricionista privado.",
    "Mi nevera tiene tantas mermeladas casi vacías que parece un laboratorio de biología experimental.",
    "Qué bien se siente viajar en bus apretado, ideal para conocer las marcas de perfume de la ciudad.",
    "Mi planta de menta se secó misteriosamente a pesar de estar en tierra abonada de primera calidad.",
    "Adoro cuando me dicen que el ambiente es dinámico y en realidad significa que salimos a las diez.",
    "Qué gran detalle de la compañía eléctrica: subieron las tarifas para incentivar el uso de velas.",
    "El asistente virtual de mi banco me bloquea la cuenta cuando intento hacer un pago de emergencia.",
    "Adoro cuando el sitio web gubernamental me pide usar un navegador de internet descontinuado.",
    "Mi gato prefiere dormir en mi cara a las tres de la mañana que en su cama térmica de lujo.",
    "Qué oportuno que el aire acondicionado falle el día que la temperatura roza los cuarenta grados.",
    "Adoro cuando me dicen que el dinero no da la felicidad personas que viajan en jet privado.",
    "Mi perro pasa horas atrapando una mosca imaginaria con una dedicación digna de un atleta.",
    "Qué bien pensado el validador del metro: da error de lectura cuando tengo a cien personas detrás.",
    "Adoro cuando la beca de investigación se cancela por entregar el papel con un minuto de retraso.",
    "Mi nevera está tan desprovista que el limón seco califica como elemento decorativo de la cocina.",
    "Qué suerte tener un despertador que suena con volumen progresivo para despertarme con infarto leve.",
    "Adoro cuando los términos de servicio tienen cien páginas de texto legal para instalar una calculadora.",
    "Mi planta de interior ha decidido marchitarse solo porque la miré fijamente durante un minuto.",
    "Qué bien diseñado el cargador del móvil: deja de funcionar si se dobla un milímetro a la izquierda."
]

# Unificar y procesar
selected_features = [
    'MeanWordLen', 'LexicalDiversity', 'MeanSentenceLen', 'StdevSentenceLen', 'DocumentLen',
    'WordsPerText', 'SentencesPerText', 'num_words', 'num_chars', 'irony_score',
    'prop_NOUN', 'prop_VERB', 'prop_ADJ', 'rhetorical_questions', 'avg_depth',
    'Flesch Score', 'Lexical Entropy', 'Syntactic Repetition', 'Unusual Word Frequency'
]

def agregar_satiras_faltantes():
    ruta_original = os.path.join("documentos_origen", "Titulacion1", "DatasetsFinales", "df_train2_featselect2.jsonl")
    
    if not os.path.exists(ruta_original):
        print(f"[ERROR] No se encontro el dataset original en {ruta_original}")
        return
        
    print(f"[LOAD] Cargando dataset expandido de {ruta_original}...")
    df_expandido = pd.read_json(ruta_original, orient='records', lines=True)
    print(f"[INFO] Dataset cargado. Contiene {len(df_expandido)} registros.")

    processor = TextProcessor()
    
    nuevos_registros = []
    print("\n[INFO] Procesando características de los 93 ejemplos de sátira cotidiana adicionales...")
    for idx, text in enumerate(ejemplos_satiricos_93):
        # Generar texto procesado
        processed_text = processor.preprocess_text(text)
        # Extraer las 19 características manuales
        features_dict = processor.calculate_features(text)
        
        # Armar el registro
        registro = {
            "id": f"refuerzo500_satirico_adicional_{idx}",
            "transcription": text,
            "transcription_processed": processed_text,
            "label": 1
        }
        
        for feat in selected_features:
            registro[feat] = features_dict.get(feat, 0)
            
        nuevos_registros.append(registro)
        
    df_nuevos = pd.DataFrame(nuevos_registros)
    print(f"[INFO] Procesados {len(df_nuevos)} nuevos registros satíricos.")
    
    # Combinar datasets (6507 + 93 = 6600)
    df_final = pd.concat([df_expandido, df_nuevos], ignore_index=True)
    
    # Guardar
    df_final.to_json(ruta_original, orient='records', lines=True)
    print(f"[SAVE] Dataset final guardado en {ruta_original} (Total: {len(df_final)} registros).")
    
    # Re-entrenar y guardar serializadores
    print("\n[INFO] Re-ajustando Vectorizador TF-IDF y Escalador MinMaxScaler...")
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_features = vectorizer.fit_transform(df_final['transcription_processed'].fillna("")).toarray()
    
    manual_features = df_final[selected_features].values
    combined_features = np.concatenate([tfidf_features, manual_features], axis=1)
    
    scaler = MinMaxScaler()
    scaler.fit(combined_features)
    
    # Guardar en static
    static_dir = os.path.join("satire_detector_api", "static")
    joblib.dump(vectorizer, os.path.join(static_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(static_dir, "minmax_scaler.pkl"))
    print(f"[INFO] tfidf_vectorizer.pkl y minmax_scaler.pkl actualizados con éxito en {static_dir}!")
    
    print("\n" + "="*60)
    print("PROCESAMIENTO DE COMPLEMENTACIÓN COMPLETADO CON ÉXITO")
    print("="*60)

if __name__ == "__main__":
    agregar_satiras_faltantes()
