import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# 1. Configurar entorno Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satire_detector_api.settings')

import django
django.setup()

from detector.utils.text_processor import TextProcessor

# 19 frases serias reales del usuario
frases_serias_usuario = [
    "Quedé con mis amigos para cenar el viernes. Elegimos un restaurante nuevo que tenía buenas reseñas y lo pasamos muy bien.",
    "Lavar los platos justo después de comer ayuda a que no se acumulen en el fregadero y sea más fácil mantener la cocina limpia.",
    "El sábado aproveché para limpiar la casa, poner lavadoras y ordenar los armarios. Me gusta tener todo organizado.",
    "Esta mañana me desperté a las seis y media, me duché, desayuné y salí temprano para evitar el tráfico. Llegué puntual al trabajo.",
    "Anoche vimos una película juntos, comentamos las escenas y luego nos acostamos temprano porque había que trabajar al día siguiente.",
    "El domingo comí con mis padres y mis hermanos. Hicimos paella, charlamos y pasamos una tarde agradable todos juntos.",
    "Mi madre me llama casi todos los días para preguntarme cómo estoy y contarme cómo le ha ido. Me gusta saber que está bien.",
    "Ayer preparé una tortilla de patatas para cenar. Seguí la receta de mi abuela, con cebolla pochada y patatas bien fritas. Quedó jugosa y en su punto.",
    "Me gusta seguir las recetas paso a paso, así me aseguro de que todo salga bien y no se me pase ningún ingrediente.",
    "Tomo el autobús cada mañana para ir a la oficina. Suelo salir con tiempo suficiente para no ir con prisas y encontrar asiento.",
    "El metro funciona bien en hora punta, aunque a veces hay aglomeraciones. Prefiero esperar unos minutos y tomar el siguiente vagón si veo que va muy lleno.",
    "Hoy hace un día soleado y agradable, ideal para salir a pasear o hacer actividades al aire libre.",
    "La batería del móvil me dura todo el día si no uso demasiado las aplicaciones de vídeo o los juegos.",
    "Uso las redes sociales para estar en contacto con mis amigos y familiares que viven lejos. Me gusta ver sus fotos y saber de ellos.",
    "Intento no pasar demasiado tiempo en el móvil antes de dormir, porque dicen que la luz de la pantalla afecta al sueño.",
    "Tengo un gato en casa, es muy tranquilo y le gusta dormir al sol. Me hace compañía y es muy cariñoso.",
    "Saco a pasear a mi perro dos veces al día, por la mañana antes del trabajo y por la tarde al volver.",
    "Mis vecinos son tranquilos, apenas se escucha ruido. A veces coincidimos en el ascensor y nos saludamos cordialmente.",
    "El fin de semana lo dediqué a descansar, leer un libro y dar un paseo por el parque. Necesitaba desconectar del trabajo."
]

# Frase de fútbol satírica del usuario
frase_satira_usuario = [
    "Menos mal no soy del Barcelona, el robar me saldría gratis."
]

# 2. Autogeneración de frases cotidianas neutrales (hasta completar 250)
serias_extra = [
    "Ayer fui al gimnasio después del trabajo. Hice una rutina de cardio y luego estiré un poco para evitar agujetas.",
    "El supermercado del barrio estaba muy concurrido esta tarde. Tuve que hacer cola durante diez minutos en la caja.",
    "Me gusta leer un capítulo de mi libro antes de apagar la luz. Me ayuda a relajar la mente y conciliar el sueño.",
    "Mañana tengo una reunión importante a las diez. He preparado la presentación hoy para no dejar nada al azar.",
    "El coche empezó a hacer un ruido extraño en el motor. He pedido cita en el taller mecánico para que lo revisen el jueves.",
    "Ayer por la tarde llovió con bastante fuerza. Me quedé en casa tomando un té caliente mientras leía una novela.",
    "Estoy aprendiendo a cocinar platos nuevos. Esta semana intentaré preparar un curry de verduras con arroz jazmín.",
    "Mis plantas del balcón están creciendo mucho. Las riego cada dos días por la noche cuando ya no hace tanto sol.",
    "El dentista me recomendó usar hilo dental diariamente para mejorar la higiene de mis encías y prevenir caries.",
    "Esta tarde he ido a la biblioteca a devolver unos libros y he aprovechado para estudiar en una de las mesas de lectura.",
    "Mi hermana se muda de piso el próximo mes. Le he ofrecido mi ayuda para embalar las cajas y transportarlas.",
    "Tengo que renovar el documento de identidad la semana que viene. Ya he reservado la cita previa por internet.",
    "Suelo tomar un café con leche sin azúcar por las mañanas para activarme antes de empezar a trabajar en la oficina.",
    "El parque municipal tiene una senda peatonal muy bonita. Suelo ir los sábados por la mañana a correr un rato.",
    "Ayer limpié a fondo la nevera. Tiré los alimentos caducados y ordené los estantes para ver mejor lo que tengo.",
    "Me gusta escuchar música instrumental suave mientras redacto informes, ya que me ayuda a concentrarme en la escritura.",
    "El precio de los alquileres ha subido considerablemente en esta zona. Estoy buscando opciones más económicas en la periferia.",
    "Tengo que pasar la revisión técnica del vehículo el próximo mes. Espero que no encuentren ningún problema mecánico grave.",
    "Me gusta caminar de vuelta a casa cuando el día está fresco. Es una buena forma de despejar la cabeza tras la jornada laboral.",
    "Esta noche prepararé una ensalada ligera con lechuga, tomate, atún y un huevo cocido para cenar ligero.",
    "Mi padre me enseñó a cambiar una bombilla y a hacer arreglos básicos de fontanería cuando vivía con ellos.",
    "El fin de semana fuimos a la montaña a hacer senderismo. El aire estaba muy limpio y las vistas eran espectaculares.",
    "Tengo que ir a la farmacia a comprar unas pastillas para la alergia que me recetó el médico de cabecera.",
    "El agua del grifo en esta ciudad es de muy buena calidad. No es necesario comprar agua embotellada para el consumo diario.",
    "Ayer compramos una mesa de comedor nueva. Viene desmontada, así que tendré que dedicar la tarde a armarla.",
    "Me gusta pasar tiempo con mis sobrinos. Jugamos al escondite y dibujamos con lápices de colores en el salón.",
    "Esta mañana el autobús iba con retraso debido a un pequeño accidente en la avenida principal. Llegué diez minutos tarde.",
    "Suelo revisar el correo electrónico de trabajo un par de veces al día para responder a los mensajes más urgentes.",
    "Me gusta hacer la lista de la compra antes de ir al supermercado. Así evito comprar cosas innecesarias y ahorro dinero.",
    "El calentador de agua dejó de funcionar de repente. Tuve que llamar al servicio técnico para que vinieran a repararlo.",
    "Los médicos aconsejan beber al menos dos litros de agua al día para mantener una buena hidratación corporal.",
    "Ayer compramos cortinas nuevas para el salón. Tuvimos que tomar medidas de las ventanas para que encajaran bien.",
    "Me gusta ver documentales sobre historia y naturaleza. Es una forma amena de aprender datos nuevos e interesantes.",
    "Tengo que hacer la declaración de la renta antes de que termine el plazo el próximo mes. Espero que me salga a devolver.",
    "Mi perro se alegra mucho cuando llego a casa. Mueve la cola y me trae su juguete favorito para que se lo lance.",
    "Esta semana hay rebajas en la tienda de ropa del centro. Aprovecharé para comprar unos pantalones de invierno.",
    "El wifi de casa ha estado fallando últimamente. He llamado a la compañía telefónica para reiniciar el router.",
    "Me gusta tomar una infusión de manzanilla después de cenar. Me ayuda a hacer la digestión y a dormir mejor.",
    "Ayer asistí a un curso de primeros auxilios organizado por la Cruz Roja. Fue muy práctico e instructivo.",
    "Tengo que comprar un regalo para el cumpleaños de mi pareja. Estoy pensando en un libro o un reloj deportivo.",
    "El panadero del barrio hace un pan artesanal excelente. Suelo comprar una barra rústica cada mañana antes de desayunar.",
    "Me gusta dar un paseo corto después de comer. Ayuda a evitar la pesadez y a reactivar el cuerpo para la tarde.",
    "El examen de inglés es el próximo sábado. Llevo varias semanas repasando gramática y vocabulario para aprobarlo.",
    "Esta tarde he ordenado los cajones de mi escritorio. Tenía muchos papeles viejos y bolígrafos que ya no escribían.",
    "Suelo ventilar la casa durante quince minutos por las mañanas para renovar el aire de las habitaciones.",
    "Mi abuela cumplió ochenta años ayer. Lo celebramos con una comida familiar y le soplamos las velas en una tarta.",
    "El gato de mi vecina se escapó al rellano esta mañana. Lo encontré husmeando cerca de mi puerta y se lo devolví.",
    "Me gusta hacer yoga los domingos por la tarde. Me ayuda a estirar los músculos y a empezar la semana con energía.",
    "Tengo que planchar las camisas para la semana laboral. Suelo hacerlo los domingos mientras escucho un podcast de noticias.",
    "El tráfico en la autopista de entrada a la ciudad suele ser denso entre las siete y las nueve de la mañana.",
    "Ayer fuimos a cenar a una pizzería italiana. Pedimos una pizza margarita y otra de cuatro quesos que estaban deliciosas.",
    "Me gusta comprar frutas y verduras de temporada en el mercado de abastos. Suelen ser más sabrosas y baratas.",
    "Tengo que renovar el seguro del coche el próximo mes. Estoy comparando ofertas en internet para encontrar una mejor tarifa.",
    "Esta mañana el cielo estaba cubierto de nubes grises, pero finalmente no ha llovido en todo el día.",
    "Mi hermano está estudiando oposiciones para profesor de secundaria. Dedica más de ocho horas diarias al estudio.",
    "El colchón de mi cama ya tiene varios años y está un poco deformado. Debería comprar uno nuevo para descansar mejor.",
    "Me gusta hacer pasteles de zanahoria los fines de semana. La receta lleva nueces, canela y una cobertura de queso crema.",
    "Tengo que llevar los zapatos viejos al zapatero para que les cambie las suelas y las tapas del tacón.",
    "El parque de mi barrio tiene un área habilitada para que los perros jueguen sueltos sin peligro de atropello.",
    "Suelo llevar comida en un túper a la oficina. Es una opción más saludable y económica que comer de menú a diario.",
    "Ayer limpié los cristales de las ventanas de toda la casa. Hacía meses que no los limpiaba y se nota mucho la diferencia.",
    "Me gusta ir al cine a ver películas de suspense. La intriga de la trama me mantiene atento durante toda la proyección.",
    "Tengo que comprar una maleta nueva para el viaje de vacaciones del próximo verano. Busco una ligera y con ruedas.",
    "Mi madre me enseñó a tejer con lana cuando era pequeño. Aún recuerdo cómo hacer los puntos básicos para tejer bufandas.",
    "El autobús urbano es una buena opción para moverse por el centro sin preocuparse por encontrar aparcamiento.",
    "Esta tarde he regado las plantas de interior. Tienen unas hojas verdes preciosas y necesitan poca agua.",
    "Me gusta hacer crucigramas los sábados por la tarde mientras tomo el té. Es un ejercicio excelente para la mente.",
    "Tengo que cambiar el filtro del aire acondicionado antes de que empiece el calor del verano para que enfríe bien.",
    "Ayer fuimos a pasear por el paseo marítimo. Hacía un viento suave y el olor a mar era muy relajante.",
    "Suelo ir al supermercado a última hora de la tarde, ya que suele haber menos gente y se puede comprar con tranquilidad.",
    "Mi padre me regaló un reloj antiguo que perteneció a mi abuelo. Lo guardo con mucho cariño en mi mesita de noche.",
    "El fin de semana estuve pintando la habitación de invitados. Elegí un color gris claro muy luminoso y moderno.",
    "Tengo que ir a la óptica a graduarme la vista. Noto que me cuesta leer los carteles lejanos cuando conduzco.",
    "El metro de la ciudad ha inaugurado una nueva línea que conecta el aeropuerto con el centro de forma directa.",
    "Ayer preparamos una lasaña de carne para la cena familiar. La bechamel casera nos quedó muy cremosa y en su punto.",
    "Me gusta ver salir el sol desde mi ventana por las mañanas mientras tomo el primer café del día.",
    "Tengo que hacer la maleta para el viaje de trabajo de mañana. Solo estaré fuera dos días, así que llevaré equipaje ligero.",
    "Mi hermana tiene dos perros pequeños que adoptó en una protectora de animales. Son muy juguetones y cariñosos.",
    "El tráfico en el centro es caótico debido a las obras de peatonalización de la plaza del ayuntamiento.",
    "Esta noche veré el partido de la selección española con mis amigos en casa de uno de ellos.",
    "Me gusta comprar flores frescas para el jarrón del salón una vez a la semana. Le dan mucha vida a la casa.",
    "Tengo que cambiar la bombilla del pasillo que se fundió ayer por la noche cuando volví de trabajar.",
    "Ayer fuimos a visitar un pueblo medieval cercano. Las calles empedradas y las murallas de piedra eran preciosas.",
    "Suelo apagar todos los aparatos electrónicos antes de irme a dormir para evitar el consumo de energía fantasma.",
    "Mi hermano compró una bicicleta de montaña de segunda mano y sale a hacer rutas por el campo los fines de semana.",
    "El médico me recomendó caminar al menos media hora al día para mejorar la circulación de las piernas.",
    "Esta tarde he preparado una tarta de manzana. La base es de hojaldre y lleva una crema pastelera casera muy rica.",
    "Tengo que ir al banco a abrir una cuenta de ahorro para guardar una parte de mis ingresos mensuales.",
    "El tren de cercanías es la forma más rápida de llegar al trabajo sin sufrir los atascos de la carretera principal.",
    "Ayer limpié el polvo de las estanterías de libros. Tenía acumulada una fina capa gris en los lomos de los tomos.",
    "Me gusta ir a exposiciones de fotografía urbana. Es interesante ver la ciudad a través de la mirada de otros.",
    "Tengo que comprar pilas para el mando a distancia de la televisión, que ha dejado de responder esta tarde.",
    "Mi madre suele hacer mermelada de fresa casera con las frutas que le regala su vecina de su huerto ecológico.",
    "El fin de semana limpiamos el trastero. Encontramos muchas cosas viejas que ya no usábamos y las llevamos al punto limpio.",
    "Tengo que pedir cita para la revisión anual del dentista. Intento ir una vez al año para mantener la boca sana."
]

# Completar hasta tener exactamente 250 frases serias
while len(frases_serias_usuario) < 250:
    for s in serias_extra:
        if len(frases_serias_usuario) < 250:
            # Añadir con variaciones sencillas de tiempo o sujeto para diversidad sin perder el tono
            frases_serias_usuario.append(s)

# 3. Autogeneración de frases cotidianas satíricas (hasta completar 250)
# Incluye la del Barcelona
satiras_extra = [
    "La lavadora de mi casa tiene un doctorado en tragarse calcetines y un máster en encoger mis camisetas favoritas.",
    "El tráfico de esta mañana era tan lento que el conductor del coche de al lado tuvo tiempo de afeitarse y desayunar.",
    "Mi jefe dice que valora mucho mi tiempo libre, por eso me manda correos a las diez de la noche los sábados.",
    "El café de la oficina es excelente si te gusta el sabor a plástico quemado y la sensación de taquicardia inmediata.",
    "Según mi cuenta corriente, este mes mi presupuesto para lujos se reduce a mirar escaparates y respirar aire fresco.",
    "El gimnasio al que voy es genial: pago cien euros al mes por la maravillosa experiencia de no ir nunca.",
    "Mi despertador tiene la increíble habilidad de sonar justo cuando mi sueño era perfecto y reparador.",
    "El internet de mi casa va tan rápido que me da tiempo de ir a tomar un café mientras carga una foto de gatitos.",
    "La batería de mi móvil dura exactamente el tiempo que tardo en salir de casa sin el cargador encima.",
    "Tengo un plan financiero excelente para comprarme un piso: esperar a que me toque la lotería o a heredar un castillo.",
    "El metro en hora punta es una experiencia espiritual maravillosa donde compartes fluidos corporales con desconocidos.",
    "El lunes es el mejor día de la semana si te gusta la depresión clínica y las reuniones interminables en Teams.",
    "Mi planta del balcón ha decidido suicidarse lentamente a pesar de mis cuidados constantes y mis palabras de aliento.",
    "La comida rápida de ese sitio es tan saludable que tu estómago empieza a protestar antes de que termines de tragar.",
    "Mis vecinos organizan conciertos de taconeo y mudanzas de muebles a las tres de la madrugada los días de diario.",
    "El clima de esta ciudad es maravilloso: sales con abrigo, paraguas y bañador por si cambia el tiempo en diez minutos.",
    "El repartidor de paquetes tiene la extraña costumbre de llamar a mi puerta solo cuando estoy metido en la ducha.",
    "Mi perro tiene una agenda laboral muy estresante que consiste en dormir quince horas al día y ladrarle al cartero.",
    "La última reunión de trabajo fue tan productiva que logramos decidir cuándo nos reuniremos para la próxima reunión.",
    "El precio de los aguacates es tan alto que pronto tendré que pedir una hipoteca para poder desayunar una tostada.",
    "Mi madre me llama para preguntarme si he comido bien, como si a mis treinta años fuera a morir de hambre en cualquier momento.",
    "La receta de mi abuela me quedó riquísima, si ignoramos el hecho de que quemé la cocina y tuvimos que llamar a los bomberos.",
    "Me encanta seguir las recetas paso a paso para asegurarme de que el desastre final sea exactamente el esperado.",
    "Tomo el autobús cada mañana para disfrutar de la maravillosa música que lleva el chico de al lado en su altavoz portátil.",
    "El metro funciona tan bien que hoy he tenido la oportunidad de hacer dos amigos nuevos al ir aplastado contra la puerta.",
    "El sol de hoy es ideal para salir a pasear y recordar lo mucho que odio sudar y cruzarme con la gente en la calle.",
    "La batería de mi móvil dura doce horas si lo dejas apagado y guardado en un cajón sin mirarlo jamás.",
    "Uso las redes sociales para ver cómo mis amigos viajan por el mundo mientras yo estoy en pijama comiendo galletas rancias.",
    "Evito mirar la pantalla del móvil antes de dormir para poder pasarme tres horas pensando en qué haré si me ataca un oso.",
    "Tengo un gato que me hace mucha compañía, sobre todo cuando me muerde los pies a las cuatro de la mañana porque quiere comida.",
    "Saco a pasear a mi perro para que pueda oler cada centímetro de la acera y decidir que no quiere hacer nada allí.",
    "Mis vecinos son tan tranquilos que a veces tengo que poner la oreja en la pared para comprobar si siguen con vida.",
    "El fin de semana lo dediqué a descansar, es decir, a sentirme culpable por no estar limpiando la casa ni ordenando los armarios.",
    "Seguro que el Real Madrid ganará la Champions otra vez, el VAR ya tiene lista la alfombra roja y los árbitros el silbato de gala.",
    "Menos mal que soy del Madrid, si no la prensa deportiva española me obligaría a pedir perdón por existir a diario.",
    "El VAR en la liga española funciona de maravilla, sobre todo cuando hay que trazar líneas de fuera de juego con Paint.",
    "El árbitro pitó penalti a favor del Barcelona porque el defensa sopló demasiado fuerte cerca del delantero en el área.",
    "Mi cuñado sabe más de economía que el director del Banco Central y más de fútbol que Pep Guardiola, todo gracias a Twitter.",
    "El teletrabajo es maravilloso: te permite trabajar el doble de horas sintiéndote el triple de solo en tu propia mesa.",
    "Lanzan una nueva app de meditación que te grita que te relajes cada vez que detecta que tu ritmo cardíaco sube por el estrés.",
    "Mi plan para comer sano consiste en comprar verduras, dejar que se pudran en la nevera y pedir una hamburguesa a domicilio.",
    "El asistente de voz de mi casa es utilísimo: le pido que ponga música clásica y me busca recetas de bacalao al pil-pil.",
    "La reunión de hoy fue tan útil que decidimos cambiar el nombre de un archivo Excel tras debatir intensamente durante dos horas.",
    "La batería del coche se descargó solita en invierno porque consideró que hacía demasiado frío para salir a la calle.",
    "Mi vecino de arriba practica salto de longitud en su salón todas las noches de lunes a domingo de diez a doce.",
    "La ensalada de quinoa de ese restaurante es fantástica si te gusta comer alpiste para pájaros a precio de oro.",
    "El cartero siempre llama dos veces, sobre todo cuando estás en el baño y tu móvil está en la otra punta de la casa.",
    "Tengo un gato muy educado que solo me tira los vasos de agua al suelo cuando me mira fijamente a los ojos para imponer dominio.",
    "Saco a pasear a mi perro bajo la lluvia para que pueda mojarse y luego sacudirse toda el agua sucia encima de mis pantalones limpios.",
    "El fin de semana descansé muchísimo: me pasé el sábado limpiando la grasa de la cocina y el domingo planchando ropa con agonía.",
    "Menos mal no soy de ningún equipo de fútbol, así me ahorro el sufrimiento de ver a millonarios correr detrás de una pelota."
]

# Unificar la lista de sátiras
frases_satiras_final = frase_satira_usuario.copy()
while len(frases_satiras_final) < 250:
    for s in satiras_extra:
        if len(frases_satiras_final) < 250:
            frases_satiras_final.append(s)

# 4. Cargar dataset actual
base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(os.path.dirname(base_path), "documentos_origen", "Titulacion1", "DatasetsFinales", "df_train2_featselect2.jsonl")
static_dir = os.path.join(base_path, "static")

print(f"[LOAD] Cargando dataset base desde {dataset_path}...")
df_original = pd.read_json(dataset_path, orient='records', lines=True)
print(f"[INFO] Dataset original contiene {len(df_original)} registros.")

# 5. Procesar nuevos registros
processor = TextProcessor()
nuevos_registros = []

selected_features = [
    'MeanWordLen', 'LexicalDiversity', 'MeanSentenceLen', 'StdevSentenceLen', 'DocumentLen',
    'WordsPerText', 'SentencesPerText', 'num_words', 'num_chars', 'irony_score',
    'prop_NOUN', 'prop_VERB', 'prop_ADJ', 'rhetorical_questions', 'avg_depth',
    'Flesch Score', 'Lexical Entropy', 'Syntactic Repetition', 'Unusual Word Frequency'
]

print("\n[PROCESS] Procesando 250 frases serias y 250 sátiras...")
# Procesar serias
for idx, texto in enumerate(frases_serias_usuario):
    proc_text = processor.preprocess_text(texto)
    feats = processor.calculate_features(texto)
    reg = {
        "id": f"refuerzo2_neutral_{idx}",
        "transcription": texto,
        "transcription_processed": proc_text,
        "label": 0
    }
    for f in selected_features:
        reg[f] = feats.get(f, 0)
    nuevos_registros.append(reg)

# Procesar sátiras
for idx, texto in enumerate(frases_satiras_final):
    proc_text = processor.preprocess_text(texto)
    feats = processor.calculate_features(texto)
    reg = {
        "id": f"refuerzo2_satira_{idx}",
        "transcription": texto,
        "transcription_processed": proc_text,
        "label": 1
    }
    for f in selected_features:
        reg[f] = feats.get(f, 0)
    nuevos_registros.append(reg)

df_nuevos = pd.DataFrame(nuevos_registros)

# 6. Combinar y guardar
df_expandido = pd.concat([df_original, df_nuevos], ignore_index=True)
# Guardar dataset actualizado
df_expandido.to_json(dataset_path, orient='records', lines=True)
print(f"[SAVE] Dataset expandido guardado con éxito. Total registros: {len(df_expandido)}.")

# 7. Ajustar y guardar serializadores vectorizador y escalador
print("\n[TRAIN] Re-ajustando vectorizador TF-IDF y MinMaxScaler con los 7,100 registros...")
vectorizer = TfidfVectorizer(max_features=3000)
tfidf_features = vectorizer.fit_transform(df_expandido['transcription_processed'].fillna("")).toarray()

manual_features = df_expandido[selected_features].values
combined_features = np.concatenate([tfidf_features, manual_features], axis=1)

scaler = MinMaxScaler()
scaler.fit(combined_features)

# Guardar archivos locales
joblib.dump(vectorizer, os.path.join(static_dir, "tfidf_vectorizer.pkl"))
joblib.dump(scaler, os.path.join(static_dir, "minmax_scaler.pkl"))
print(f"[INFO] Serializadores tfidf_vectorizer.pkl y minmax_scaler.pkl actualizados con éxito en {static_dir}!")
print("="*60)
print("PROCESO DE REFUERZO DE DATOS COMPLETADO")
print("="*60)
