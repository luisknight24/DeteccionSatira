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

# 2. Definición del dataset de refuerzo cotidiano (250 serios, 250 satíricos)
ejemplos_serios = [
    # 28 Ejemplos provistos por el usuario
    "Yo pienso que es importante escuchar a las personas, conocer sus opiniones, sus ideas, es parte de la convivencia diaria.",
    "Si formaramos parte de una sociedad, debe haber respeto y comunicación.",
    "Ayer fui a la universidad y entregué mis tareas, el profesor me calificó con 10 y resaltó mi esfuerzo.",
    "Es complicado ser docente, algunas veces los estudiantes son irrespetuosos y no escuchan las indicaciones, quieren causar conflictos y portarse mal, aunque no tengan la razón.",
    "A veces pienso que nos falta empatía, nos cuesta ponernos en el lugar del otro antes de juzgar sus acciones o sus palabras.",
    "Me gusta dedicar tiempo a leer, aunque sea un rato cada noche. Me ayuda a desconectar del día y a aprender cosas nuevas sin darme cuenta.",
    "Pienso que viajar, aunque sea a lugares cercanos, te abre la mente y te enseña a ver las cosas desde otra perspectiva.",
    "No todo en la vida es trabajo, también hay que saber disfrutar los pequeños momentos, como una conversación con un amigo o una tarde tranquila en casa.",
    "Ayer fui al supermercado y noté que varios productos habían subido de precio. Compré solo lo necesario y dejé algunas cosas para después.",
    "El sábado salí a caminar por el parque con mi perro, hacía buen tiempo y había mucha gente paseando, niños jugando y señores haciendo ejercicio.",
    "Esta mañana se me hizo tarde para el trabajo porque el despertador no sonó. Al final llegué justo a tiempo, pero pasé un mal rato.",
    "Fui a visitar a mi abuela el domingo, le llevé un bizcocho que hice en casa y estuvimos conversando toda la tarde. Siempre me cuenta historias de cuando era joven.",
    "Ayer me junté con unos amigos a cenar, pedimos pizza y estuvimos hablando de nuestras cosas hasta tarde. Hacía tiempo que no nos veíamos todos juntos.",
    "Hoy en el trabajo tuve una reunión larguísima, pero al menos pudimos resolver varios temas pendientes que llevaban semanas sin decidirse.",
    "Mi compañera de oficina es muy amable, siempre está dispuesta a ayudar cuando alguien no entiende algo o tiene mucha carga de trabajo.",
    "Me costó bastante aprobar esa materia, tuve que estudiar muchas horas y hacer ejercicios de práctica hasta que por fin entendí los conceptos.",
    "A veces el trabajo puede ser estresante, sobre todo cuando se acumulan las tareas y los plazos de entrega son ajustados.",
    "Me gusta mi trabajo, pero creo que podría mejorar si hubiera mejor comunicación entre los equipos y menos reuniones innecesarias.",
    "Mi madre siempre me dice que hay que tratar a los demás como nos gustaría que nos trataran a nosotros, y creo que tiene toda la razón.",
    "Hablar con mis hermanos por teléfono me alegra el día, aunque sea una llamada corta para saber cómo están y contarles cómo me va.",
    "Los domingos en casa de mis padres son sagrados, nos reunimos a comer, charlamos de todo un poco y luego vemos alguna película juntos.",
    "Mi pareja y yo tuvimos una discusión tonta el otro día, pero lo hablamos, nos disculpamos y quedó todo bien.",
    "Criar a un hijo no es nada fácil, hay días en que te sientes agotada, pero verlo crecer y aprender cosas nuevas lo compensa todo.",
    "Me encanta cocinar los fines de semana, probar recetas nuevas y compartir la comida con mi familia o mis amigos.",
    "Últimamente me he aficionado a las plantas, tengo varias en el balcón y me relaja mucho regarlas y ver cómo crecen.",
    "Disfruto mucho escuchando música mientras hago las tareas de la casa, me hace sentir más animada y se me pasa el tiempo volando.",
    "Me gusta salir a correr temprano por la mañana, cuando todavía no hay mucha gente en la calle y se respira aire fresco.",
    "Ver series con mi pareja es uno de mis planes favoritos, nos ponemos cómodos en el sofá y nos olvidamos del mundo por un rato.",

    # 222 Ejemplos adicionales cotidianos / serios / no-satíricos
    "Ayer terminé el libro que me prestaste y me pareció muy interesante, sobre todo la perspectiva histórica que tiene.",
    "El próximo lunes empezaré el curso de inglés que tanto quería hacer para mejorar mis oportunidades laborales.",
    "Hoy estuvo lloviendo toda la tarde, así que me quedé en casa ordenando mis cosas y preparando café caliente.",
    "Mi abuelo siempre me decía que la paciencia es la clave para resolver cualquier malentendido familiar.",
    "Mañana tenemos una videollamada de equipo para planificar las actividades y entregas del próximo mes.",
    "Me gusta salir a caminar por el parque cuando cae la tarde, me ayuda a despejar la mente después del estudio.",
    "Fui al consultorio médico para mi chequeo anual y por suerte todo salió excelente en los análisis.",
    "El fin de semana acomodé mi armario y separé la ropa que ya no utilizaba para donarla a un albergue.",
    "Mi hermana menor está estudiando medicina y pasa casi todo el día repasando sus apuntes de anatomía.",
    "Hoy preparamos una cena especial en casa para celebrar que mi primo consiguió su primer empleo formal.",
    "Es importante mantener una rutina de ejercicio regular para cuidar nuestra salud física y mejorar el ánimo.",
    "Ayer se cortó la energía eléctrica en mi barrio por unas horas, así que aprovechamos para conversar en la sala.",
    "Me encanta caminar por la playa en invierno, cuando el lugar está vacío y solo se escucha el sonido del mar.",
    "Este fin de semana iré a visitar a mis tíos al campo, tienen un huerto muy grande y siempre nos regalan verduras frescas.",
    "Hoy tuvimos que resolver un problema técnico en la oficina que nos tomó toda la mañana de trabajo coordinado.",
    "La comunicación transparente entre los miembros de un equipo es la base para lograr terminar los proyectos a tiempo.",
    "El profesor explicó el tema de derivadas de manera muy clara, lo que nos facilitó realizar los ejercicios prácticos.",
    "Me compré unas macetas nuevas para el balcón y sembré algunas semillas de albahaca y menta para cocinar.",
    "Ayer vi una película documental sobre la vida marina en el océano Pacífico que me dejó muy impresionado.",
    "Es agradable levantarse temprano el domingo y disfrutar de la tranquilidad antes de que todos comiencen a despertar.",
    "Mi perro pasa la mayor parte de la tarde durmiendo en la alfombra de la sala donde le llega un poco de sol.",
    "Esta semana ha sido muy intensa en la oficina, pero logramos entregar todos los reportes solicitados por la gerencia.",
    "Disfruto mucho de las charlas tranquilas los viernes por la tarde después de terminar las responsabilidades diarias.",
    "El agua potable es un recurso vital que debemos cuidar y utilizar de forma consciente en nuestros hogares.",
    "Decidí aprender a tocar el teclado, aunque al principio me resulta difícil coordinar los movimientos de ambas manos.",
    "Ayer me encontré con un antiguo compañero del colegio y nos quedamos platicando de nuestras vidas actuales.",
    "Mi madre me enseñó una receta tradicional para hacer empanadas que siempre prepara en las reuniones familiares.",
    "Es fundamental respetar las normas de tránsito al conducir por la ciudad para garantizar la seguridad de los peatones.",
    "Hoy fui a la biblioteca municipal a buscar información para redactar el marco teórico de mi proyecto.",
    "Me gusta escuchar el sonido de la lluvia en la ventana mientras leo un libro en la sala de estar.",
    "El próximo mes viajaré a otra ciudad para asistir al matrimonio de una de mis mejores amigas de la infancia.",
    "Ayer dediqué varias horas a preparar diferentes porciones de comida para congelar y tener listas para la semana.",
    "El respeto a las opiniones diferentes es la base fundamental para mantener una convivencia pacífica en la comunidad.",
    "Me parece muy valioso dedicar un momento del día a estar en silencio y pensar en los objetivos personales.",
    "Hoy salí a montar bicicleta y recorrí casi quince kilómetros por la ruta establecida para ciclistas.",
    "Mi compañera de cuarto me ayudó a pintar una de las paredes de la sala que estaba bastante desgastada.",
    "La constancia nos ayuda a superar los obstáculos difíciles y a adquirir nuevas habilidades profesionales.",
    "Es reconfortante ver cómo los alumnos se esfuerzan por comprender temas complejos de matemáticas y programación.",
    "Hoy horneé un pastel de manzana siguiendo un tutorial de internet y me quedó muy suave y sabroso.",
    "Es importante ventilar los espacios del hogar diariamente para renovar el aire y evitar la acumulación de humedad.",
    "He decidido organizar mejor mi tiempo diario, asignando horas específicas para estudiar y momentos para descansar.",
    "El servicio de internet estuvo fallando toda la tarde, por lo que tuve que usar los datos de mi teléfono móvil.",
    "Mi padre me ayudó a revisar el motor de mi coche porque hacía un ruido extraño al encender el aire acondicionado.",
    "Pasar tiempo de calidad con las personas que queremos es la mejor manera de recargar energías el fin de semana.",
    "Hoy asistí a una clase de yoga para principiantes y me sentí muy relajado al finalizar la sesión.",
    "Es necesario revisar con atención los contratos antes de firmar para estar seguros de todas las cláusulas establecidas.",
    "Ayer limpiamos a fondo la cocina y ordenamos la alacena, descartando los productos que estaban vencidos.",
    "Me gusta ir a la feria local los sábados por la mañana a comprar frutas frescas directamente de los productores.",
    "Mi hermano adoptó un gato de la calle que es muy travieso y siempre quiere jugar con los cordones de los zapatos.",
    "La educación vial debería enseñarse desde la escuela primaria para generar una cultura de respeto en las calles.",
    "Hoy completé todos los trámites para renovar mi licencia de conducir y por fin me la entregaron por la tarde.",
    "Me resulta relajante pintar con acuarelas los fines de semana, aunque no sea un experto en el dibujo artístico.",
    "Ayer tuvimos una cena de bienvenida para un nuevo compañero de trabajo que se integró al área de sistemas.",
    "Es indispensable lavarse las manos con frecuencia para prevenir enfermedades y mantener una higiene adecuada.",
    "Me gusta escuchar programas de radio sobre divulgación científica e historia mientras hago la limpieza de la casa.",
    "El próximo fin de semana habrá una jornada de reforestación en el cerro local y me inscribí como voluntario.",
    "Hoy recibí el paquete que había pedido por internet hace dos semanas, llegó en perfectas condiciones y a tiempo.",
    "Estudiar un nuevo idioma requiere mucha práctica oral diaria y no tener miedo a cometer errores de pronunciación.",
    "Mi tía prepara un postre helado de limón que es ideal para los días calurosos del verano.",
    "Es importante cuidar la postura al trabajar frente a la computadora para evitar dolores de espalda al final de la jornada.",
    "Hoy fui al banco a abrir una cuenta de ahorros para empezar a destinar fondos a mis planes de especialización.",
    "Ayer ayudé a mi vecino a mover unos muebles pesados porque está haciendo remodelaciones en su sala.",
    "El parque del centro tiene árboles muy antiguos y es un lugar excelente para sentarse a leer en las mañanas.",
    "Es recomendable realizar copias de seguridad de nuestros archivos importantes con regularidad para evitar pérdidas.",
    "Hoy me levanté temprano para ver el amanecer y tomar fotografías de la ciudad antes de que comience el tráfico.",
    "El curso de fotografía digital me ha enseñado a manejar mejor los tiempos de exposición y la iluminación natural.",
    "Mi abuela me tejió una bufanda de lana muy abrigadora para el invierno que ya empezó a sentirse con fuerza.",
    "La honestidad y el respeto mutuo son valores indispensables que debemos fomentar en todas nuestras relaciones sociales.",
    "Hoy preparé una ensalada mixta con lechuga, tomate, aguacate y pollo a la plancha para almorzar ligero.",
    "Es muy útil aprender nociones básicas de primeros auxilios para saber cómo reaccionar en casos de emergencia.",
    "El autobús de regreso a casa tardó más de lo habitual debido a unos trabajos de pavimentación en la avenida principal.",
    "Hoy logré terminar de redactar la introducción de mi informe y espero avanzar con el desarrollo mañana.",
    "Me gusta la música instrumental suave cuando tengo que concentrarme en tareas detalladas de programación.",
    "Este año decidí reducir el consumo de plástico de un solo uso, utilizando botellas y bolsas reutilizables.",
    "Ayer visité el museo de historia de la ciudad y aprendí sobre la fundación y las primeras construcciones locales.",
    "Mantener el espacio de trabajo limpio y ordenado favorece la concentración y reduce el estrés laboral.",
    "Hoy mi jefe reconoció el esfuerzo del equipo en la entrega del proyecto ante los directores de la empresa.",
    "Aprendí a programar en Python gracias a una serie de cursos gratuitos disponibles en plataformas educativas virtuales.",
    "Es saludable tomar al menos dos litros de agua al día para mantener una buena hidratación general.",
    "El próximo sábado me reuniré con mis primos para jugar juegos de mesa y compartir una tarde tranquila en familia.",
    "Ayer se celebró la feria del libro en la plaza principal y compré una novela histórica de un autor local.",
    "Es fundamental descansar las horas necesarias para tener un buen rendimiento en el estudio y el trabajo.",
    "Hoy preparé un puré de papas con pescado al horno para cenar y me quedó muy rico y sencillo.",
    "La paciencia y la empatía son herramientas clave para resolver diferencias de opinión en el ámbito familiar.",
    "El clima ha estado muy variable últimamente, con mañanas frías y tardes bastante calurosas.",
    "Hoy asistí a una charla informativa sobre becas de posgrado en el extranjero y tomé nota de los requisitos.",
    "Ayer ayudé a mi madre a ordenar las plantas del jardín y a trasplantar algunas flores a macetas más grandes.",
    "Es aconsejable apagar los dispositivos electrónicos al menos treinta minutos antes de ir a dormir para descansar mejor.",
    "Hoy caminé de regreso a casa en lugar de tomar el autobús para hacer un poco de ejercicio diario.",
    "El mantenimiento oportuno de los electrodomésticos ayuda a prolongar su vida útil y a ahorrar energía en el hogar.",
    "Hoy el cielo amaneció completamente despejado y con un sol muy brillante que anima el inicio de la jornada.",
    "Comencé a hacer un curso en línea sobre finanzas personales para aprender a administrar mejor mi presupuesto mensual.",
    "Mi hermana me regaló un cuadro pintado por ella misma para decorar la pared vacía de mi habitación.",
    "El trabajo colaborativo enriquece las soluciones a los problemas porque aporta diversas perspectivas y experiencias.",
    "Ayer compramos un escritorio nuevo para armar una zona de estudio cómoda y bien iluminada en la casa.",
    "Es saludable establecer límites claros entre el tiempo de trabajo y la vida familiar al realizar teletrabajo.",
    "Hoy preparé jugo de naranja natural por la mañana y lo acompañé con unas tostadas con queso integral.",
    "La perseverancia en el estudio de las matemáticas rinde frutos cuando por fin se logran resolver los problemas solos.",
    "El autobús en el que viajo tiene asientos cómodos y suelo aprovechar el recorrido para escuchar audiolibros interesantes.",
    "Ayer por la tarde salimos a caminar en familia y compramos helado en una pequeña tienda del vecindario.",
    "Es muy importante mantener la calma y actuar con prudencia ante situaciones imprevistas o estresantes.",
    "Hoy finalicé de leer un libro sobre la historia del arte medieval que me pareció muy instructivo.",
    "El médico me recomendó incluir más verduras de hoja verde y legumbres en mi alimentación habitual.",
    "Ayer ordené mi escritorio y clasifiqué los documentos importantes en carpetas para encontrarlos rápidamente.",
    "Es agradable escuchar el cantar de los pájaros en el jardín por la mañana mientras tomo mi café diario.",
    "Este fin de semana tenemos planeado pintar la fachada de la casa en familia para darle un mejor aspecto.",
    "Hoy asistí a un seminario web sobre las tendencias de desarrollo web y las nuevas librerías de JavaScript.",
    "Mi hermano me enseñó a preparar una salsa de tomate casera muy sabrosa para acompañar las pastas del domingo.",
    "La tolerancia y el respeto a la diversidad son pilares fundamentales para una sociedad justa y armoniosa.",
    "Hoy fui al supermercado temprano y estaba muy tranquilo, por lo que pude hacer las compras con calma.",
    "Es recomendable realizar pausas activas cada dos horas de trabajo frente al computador para evitar contracturas.",
    "Ayer instalamos una repisa en el pasillo para ordenar los libros que ya no cabían en la biblioteca.",
    "La lectura nocturna es una excelente forma de relajar la mente y desconectarse de las obligaciones diarias.",
    "Hoy preparamos panqueques de avena y plátano para el desayuno y les agregamos un poco de miel de abeja.",
    "Es fundamental cuidar las fuentes de agua dulce para garantizar el abastecimiento de las futuras generaciones.",
    "El fin de semana pasado visitamos una reserva natural y pudimos ver varias especies de aves locales en su hábitat.",
    "Hoy completé la entrega del último informe del semestre y me siento muy aliviado por haber terminado a tiempo.",
    "Mi madre me regaló unas plantas colgantes que colocamos en el balcón y le dan mucha vida al departamento.",
    "El diálogo respetuoso es el camino más efectivo para solucionar los conflictos en cualquier grupo humano.",
    "Hoy salí a caminar por la avenida principal y noté que abrieron una nueva librería con cafetería integrada.",
    "Es saludable evitar el consumo excesivo de sal y alimentos ultraprocesados para proteger la salud arterial.",
    "Ayer ayudé a mi primo a repasar conceptos de álgebra para su examen de ingreso a la facultad de ingeniería.",
    "La luz natural en el espacio de estudio mejora la concentración y reduce la fatiga visual al leer.",
    "Hoy el despertador funcionó correctamente y pude desayunar tranquilo antes de salir al trabajo.",
    "El taller de carpintería me ha permitido fabricar mis propias repisas de madera para ordenar el taller.",
    "Ayer conversé por teléfono con un amigo que vive en el extranjero y estuvimos al día con nuestras noticias.",
    "Es aconsejable revisar periódicamente el estado de las llantas del coche antes de realizar un viaje largo.",
    "Hoy preparé una sopa de verduras caliente porque el clima ha estado bastante frío y lluvioso en la ciudad.",
    "La disciplina diaria en la práctica de un instrumento es lo que permite avanzar y tocar piezas más complejas.",
    "El parque de mi vecindario cuenta con áreas verdes amplias donde la gente suele ir a hacer ejercicio.",
    "Ayer se celebró una campaña de limpieza en el parque del barrio y muchos vecinos nos sumamos a colaborar.",
    "Es importante mantener una actitud positiva y constructiva frente a los desafíos que se presentan a diario.",
    "Hoy preparé té verde con limón y jengibre para tomar por la tarde y mantenerme abrigado.",
    "La empatía nos permite comprender mejor las dificultades por las que pasan los demás y ofrecer ayuda sincera.",
    "El transporte público de la ciudad ha incorporado nuevos buses eléctricos que contaminan menos el ambiente.",
    "Hoy finalicé un curso breve de primeros auxilios y aprendí técnicas valiosas para actuar en accidentes.",
    "Ayer ordenamos el garaje de la casa y reciclamos varias cajas de cartón y botellas de plástico acumuladas.",
    "Es fundamental fomentar el hábito de la lectura en los niños mediante libros adecuados a su edad y gustos.",
    "Hoy el almuerzo estuvo delicioso, comimos arroz integral con pollo guisado y una ensalada de tomate y pepino.",
    "Es recomendable proteger la piel del sol utilizando protector solar diariamente, incluso en días nublados.",
    "Ayer acompañé a mi hermana a comprar sus materiales escolares para el inicio de clases en la universidad.",
    "La ventilación de los espacios de estudio favorece una mejor oxigenación y ayuda a mantener la mente despierta.",
    "Hoy salí a caminar temprano y el aire se sentía sumamente fresco y revitalizante antes del inicio del tráfico.",
    "El curso de diseño gráfico me ha permitido aprender a estructurar los elementos visuales de manera balanceada.",
    "Mi abuela me regaló un recetario escrito a mano con todas sus comidas favoritas para que aprenda a cocinarlas.",
    "La solidaridad entre los vecinos es fundamental para mejorar la seguridad y la convivencia en nuestra calle.",
    "Hoy preparé una avena tibia con manzana y canela para desayunar antes de iniciar las clases.",
    "Es importante realizarse controles médicos preventivos una vez al año para detectar cualquier anomalía a tiempo.",
    "El autobús de hoy venía bastante cómodo, por lo que pude leer varios capítulos de mi novela favorita.",
    "Ayer por la tarde fuimos a dar una vuelta por el centro comercial y compramos unos zapatos nuevos.",
    "Es necesario actuar con prudencia y paciencia al enfrentar situaciones estresantes o conflictos interpersonales.",
    "Hoy terminé de escribir el resumen ejecutivo del informe de investigación y quedó bastante estructurado.",
    "El médico me aconsejó caminar al menos treinta minutos diarios para fortalecer mi salud cardiovascular.",
    "Ayer limpiamos las ventanas de la casa y la entrada, lo que mejoró mucho la iluminación natural en el interior.",
    "Es agradable levantarse con tiempo de sobra para poder disfrutar de un desayuno completo en familia.",
    "El próximo fin de semana iremos a acampar a un parque nacional cercano para desconectar de la rutina de la ciudad.",
    "Hoy asistí a una conferencia en línea sobre el uso de energías renovables en las zonas rurales del país.",
    "Mi hermano preparó una pasta casera espectacular con salsa boloñesa para el almuerzo del domingo.",
    "La tolerancia es un valor fundamental para construir relaciones sanas y convivir pacíficamente con los demás.",
    "Hoy fui al supermercado en la tarde y compré verduras y frutas frescas para toda la semana.",
    "Es importante hacer pausas para estirar el cuerpo durante las jornadas de estudio prolongadas frente a la mesa.",
    "Ayer colocamos un perchero en la entrada de la casa para colgar los abrigos y bolsos de manera ordenada.",
    "La lectura de novelas es un excelente método para estimular la imaginación y ampliar el vocabulario en español.",
    "Hoy desayuné yogur natural con frutos rojos y nueces, una combinación muy fresca y saludable.",
    "Es vital promover el uso racional del agua y evitar pérdidas en los grifos y tuberías de la casa.",
    "El fin de semana visitamos una pequeña reserva botánica y tomamos apuntes sobre las flores autóctonas.",
    "Hoy entregué a tiempo el informe final de mi proyecto de investigación y me siento sumamente satisfecho.",
    "Mi madre colocó unas macetas con flores hermosas en la ventana y alegran mucho la vista desde la sala.",
    "El diálogo constructivo es siempre la mejor herramienta para resolver diferencias de criterio entre compañeros.",
    "Hoy caminé por el parque y me senté un rato a observar a las personas que paseaban a sus mascotas.",
    "Es aconsejable reducir el consumo de azúcares y grasas saturadas para mantener un buen estado físico.",
    "Ayer ayudé a mi hermana a estudiar los temas de química orgánica para su examen parcial universitario.",
    "La iluminación natural favorece un mejor rendimiento escolar y disminuye el cansancio visual del estudiante.",
    "Hoy el tráfico estuvo muy tranquilo por la mañana y logré llegar a mi destino antes de lo programado.",
    "El taller de costura me ha permitido arreglar varias prendas de vestir que tenía guardadas sin usar.",
    "Ayer llamé a mi abuelo para saber cómo se encontraba y estuvimos conversando un buen rato por teléfono.",
    "Es importante revisar el nivel de aceite del vehículo de manera periódica para evitar fallas mecánicas.",
    "Hoy preparé una sopa de lentejas caliente para el almuerzo porque el día se tornó bastante fresco.",
    "La constancia en la práctica deportiva es lo que permite mejorar la resistencia y la condición general.",
    "El parque de la esquina tiene una pista de trote muy cómoda a la que suelo ir por las mañanas.",
    "Ayer participamos en una jornada comunitaria para pintar las bancas y juegos del parque infantil del barrio.",
    "Es fundamental mantener un enfoque constructivo y optimista al enfrentarse a problemas cotidianos.",
    "Hoy tomé una taza de té de manzanilla caliente por la noche para relajarme antes de dormir.",
    "La empatía nos ayuda a comprender mejor el punto de vista del otro y evitar discusiones estériles.",
    "El transporte de la ciudad ha mejorado sus rutas para reducir los tiempos de viaje de la población.",
    "Hoy hice un curso corto sobre prevención de incendios y aprendí cómo utilizar correctamente un extintor.",
    "Ayer organizamos el armario de herramientas y clasificamos los tornillos y clavos por tamaño.",
    "Es de suma importancia promover el hábito de la lectura en los jóvenes mediante formatos que capten su atención.",
    "Hoy almorcé arroz blanco con frijoles y carne deshebrada, una comida muy clásica y balanceada.",
    "Es conveniente protegerse de la radiación solar usando sombrero y anteojos oscuros en las horas pico.",
    "Ayer acompañé a mi mamá a realizar unos trámites notariales en el centro y todo se resolvió rápido.",
    "La ventilación adecuada en los espacios de oficina es clave para el bienestar y la concentración del personal.",
    "Hoy desperté antes de que sonara la alarma y me sentí muy descansado y con buena energía para empezar.",
    "El curso de diseño vectorial me ha dotado de habilidades para crear logotipos e ilustraciones de forma profesional.",
    "Mi tía me obsequió un libro de cuentos tradicionales que leía cuando era niña para que los conserve.",
    "La cooperación entre todos los vecinos es fundamental para mantener la limpieza y el orden en nuestro pasaje.",
    "Hoy preparé unos huevos revueltos con espinacas y cebolla para desayunar antes de salir al trabajo.",
    "Es aconsejable realizarse análisis de sangre preventivos de manera anual para vigilar los niveles de salud.",
    "El tren en el que viajo hoy venía muy silencioso, así que aproveché para avanzar en la lectura de mi libro.",
    "Ayer por la tarde salimos a dar un paseo por la plaza de armas y compramos unos dulces artesanales.",
    "Es de vital importancia mantener la compostura y actuar con mesura ante eventos inesperados o tensos.",
    "Hoy terminé de estructurar el cronograma de actividades de mi propuesta de tesis y quedó muy detallado.",
    "El médico familiar me recomendó hacer caminatas de treinta minutos diarios para bajar los niveles de estrés.",
    "Ayer limpiamos la cochera y donamos varias cajas con libros y juguetes antiguos que ya no utilizábamos.",
    "Es muy satisfactorio levantarse temprano y tener tiempo para planificar las tareas del día con calma.",
    "El próximo mes iremos a visitar a unos familiares que viven cerca de un lago precioso en el sur del país.",
    "Hoy asistí a una ponencia sobre el uso de tecnologías móviles en el desarrollo de la agricultura orgánica."
]

ejemplos_satiricos = [
    # 250 Ejemplos de sátira cotidiana / redes sociales / tweets / parodia laboral
    "Adoro las reuniones de tres horas que pudieron haber sido un correo de dos líneas, se aprende mucho sobre decoración de salas.",
    "Mi jefe dice que somos como una gran familia, lo cual tiene sentido porque la mitad no nos hablamos y la otra mitad conspira en secreto.",
    "Me encanta trabajar los fines de semana de forma voluntaria obligatoria, mi vida social estaba demasiado sobrevalorada de todos modos.",
    "El tráfico hoy estaba tan fluido que solo tardé dos horas en avanzar tres calles, una velocidad de crucero envidiable.",
    "Qué alegría despertarse a las cinco de la mañana gracias al taladro de mi vecino, seguro está haciendo arte contemporáneo en su pared.",
    "Fui al supermercado y compré dos tomates y un limón, ahora estoy buscando asesoría financiera para poder pagar el préstamo de las compras.",
    "El despertador de mi móvil es mi mejor amigo, siempre se asegura de despertarme con un susto de muerte para empezar el día con el cardio al máximo.",
    "Mi planta favorita de interior ha decidido suicidarse hoy porque olvidé regarla por dos días, se ve que no soporta mi estilo de vida.",
    "Ahorrando cinco euros al mes, calculo que para el año 2200 ya tendré lo suficiente para dar el enganche de una cochera pequeña.",
    "Me encanta cuando el transporte público viene tan lleno que no necesito sujetarme de las barras porque la masa humana me sostiene.",
    "Mi compañera de piso es un amor, siempre deja sus platos en el fregadero para que yo practique mis habilidades de lavado de vajillas.",
    "Es maravilloso cómo mi conexión de internet decide irse a descansar justo cuando tengo una entrega de trabajo urgente, muy empática.",
    "Qué gran idea poner el examen de admisión un domingo a las siete de la mañana, ideal para evaluar nuestra capacidad de supervivencia zombie.",
    "El robot aspirador inteligente se ha quedado atascado bajo la cama otra vez, se ve que su inteligencia artificial prefiere la comodidad del polvo.",
    "Mi banco me envió un correo felicitándome por mis finanzas saludables, justo después de cobrarme quince euros por mantenimiento de cuenta.",
    "Me fascina cuando la gente dice 'llegando en cinco minutos' y en realidad están saliendo de la ducha en otra zona de la ciudad.",
    "El manual de productividad me aconseja levantarme a las cuatro de la mañana, meditar, hacer ejercicio y leer antes de trabajar. Yo preferí seguir durmiendo.",
    "Qué bien se siente comer ensalada de lechuga y agua mientras observas cómo la pizza del refrigerador te guiña un ojo de forma seductora.",
    "Mi gato me mira con un desprecio tan profundo que a veces me pregunto si yo soy el dueño de la casa o simplemente su sirviente sin sueldo.",
    "El ayuntamiento va a instalar ciclovías aéreas, ideal para los ciclistas que siempre quisieron experimentar el vuelo libre sin red de seguridad.",
    "Adoro cuando me dicen 'tenemos que hablar' y me dejan esperando doce horas para decirme qué marca de leche comprar.",
    "Mi carrera profesional avanza a pasos agigantados, hoy me ascendieron a 'empleado que hace el trabajo de tres personas por el mismo sueldo'.",
    "Qué gran noticia que suban los precios de la gasolina, así tengo la excusa perfecta para hacer senderismo obligatorio hasta mi oficina.",
    "El manual de autoayuda dice que debo visualizar el éxito. Llevo tres horas visualizando un saco de dinero pero solo veo pelusas en mi cuarto.",
    "Qué generosos son en mi trabajo, nos regalaron un bolígrafo con el logo de la empresa para compensar las horas extras no pagadas de este mes.",
    "Es increíble cómo mi cuerpo procesa el café de las tres de la mañana convirtiéndolo en reflexiones profundas sobre mis malas decisiones de la escuela.",
    "Mi balcón ahora tiene tres plantas secas que juré cuidar, un hermoso cementerio botánico que refleja mi gran sentido de la responsabilidad.",
    "Qué plan tan divertido pasar la tarde del sábado haciendo trámites gubernamentales en una página web que parece diseñada en el año mil novecientos noventa y cinco.",
    "El transporte público ha decidido implementar el viaje interactivo: si logras entrar al vagón, ganas el derecho a respirar aire compartido.",
    "Mi banco se preocupa tanto por mí que me cobra una comisión por retirar mi propio dinero en un cajero automático de su propia red.",
    "Adoro cuando se corta la luz y puedo disfrutar del silencio absoluto de mis dispositivos sin batería mientras lloro en voz baja.",
    "Qué bien funciona el asistente de voz de mi teléfono, le pido que me ponga música suave y decide llamar a mi jefe a las once de la noche.",
    "La dieta de comer aire purificado y agua mineral está dando resultados increíbles: ya he perdido la mitad de mis ahorros en consultas de nutrición.",
    "Qué suerte tengo de vivir en un apartamento tipo estudio de quince metros cuadrados, así puedo cocinar, dormir y lavar la ropa sin dar un solo paso.",
    "Mi plan de ahorro para comprar casa consiste en esperar a ganar la lotería de un país en el que ni siquiera he comprado un boleto.",
    "Me encanta cuando mi jefe me envía un mensaje los domingos diciendo 'no es urgente' pero añade un signo de exclamación al final de cada frase.",
    "El gimnasio de mi barrio está muy bien equipado, sobre todo la zona donde pagas la membresía anual y nunca vuelves a pararte por ahí.",
    "Qué emocionante es el proceso de actualización de software, nunca sabes si tu ordenador volverá a encender o se convertirá en un costoso pisapapeles.",
    "Adoro cuando me dan consejos financieros personas que heredaron tres propiedades de sus abuelos y viven de las rentas.",
    "Mi gato ha aprendido a abrir la puerta del baño justo cuando tengo visitas en casa, un anfitrión excelente y muy educado.",
    "Qué gran avance la inteligencia artificial, ahora los robots pueden escribir poemas horribles mientras los humanos seguimos lavando los platos a mano.",
    "Mi plan de ejercicio matutino consiste en apagar la alarma del teléfono cinco veces seguidas utilizando diferentes dedos para no cansarme.",
    "Qué oportuno que el supermercado decida cambiar de pasillo todos los productos justo el día en que tengo prisa por hacer las compras.",
    "Adoro cuando me dicen que sea proactivo y luego me regañan por tomar decisiones sin consultar a los catorce directores de área.",
    "Mi nevera está tan vacía que la luz interior ahora sirve para iluminar la profunda soledad y la falta de presupuesto de mi cocina.",
    "Qué gran idea obligarnos a usar contraseñas con mayúsculas, números, símbolos y el nombre de nuestra primera mascota en arameo para ver el recibo de la luz.",
    "El servicio al cliente de mi compañía telefónica es excelente: te atienden en cinco minutos de espera que en realidad duran cuarenta y cinco.",
    "Qué bien se viaja en hora pico en el metro, es como un abrazo grupal obligatorio de trescientas personas que no conoces de nada.",
    "Mi planta de albahaca ha decidido morir porque la miré de forma un poco brusca esta mañana, qué delicadeza de ser vivo.",
    "Adoro los correos electrónicos corporativos que empiezan con 'espero que este mensaje te encuentre bien' en medio de una crisis de servidores general.",
    "Mi jefe me pidió que fuera transparente, así que le mostré la cara de desesperación que pongo cada vez que me asigna una nueva tarea.",
    "Qué divertido es pasar las vacaciones en casa viendo fotos de mis amigos en playas paradisíacas mientras yo peleo con las hormigas de mi cocina.",
    "El banco me envió una tarjeta de crédito preaprobada con un límite de saldo equivalente a tres meses de mi sueldo, para que aprenda a endeudarme con estilo.",
    "Adoro cuando me dicen 'sé tu propio jefe' y resulta que solo significa trabajar catorce horas diarias sin seguro social ni vacaciones pagadas.",
    "Qué gran invento las videollamadas de trabajo donde puedes ver la cara de aburrimiento de tus compañeros en alta definición y desde diferentes ángulos.",
    "Mi gato se ha adueñado de mi silla de trabajo y ahora tengo que hacer mis labores de rodillas en el suelo para no incomodar a su majestad.",
    "Qué oportuno que el neumático de mi coche decida desinflarse justo cuando está lloviendo y me quedan diez minutos para llegar a la entrevista.",
    "El curso de finanzas me enseñó que si dejo de comprar café en la calle durante ochenta años, podré comprar un metro cuadrado de terreno.",
    "Adoro cuando la página del gobierno se cae a mitad del trámite y tengo que volver a subir los veintidós documentos digitalizados en formato PDF de un mega.",
    "Qué bien se siente pagar impuestos para que luego repavimenten la misma calle tres veces en un mes mientras las demás avenidas parecen la superficie lunar.",
    "Mi plan de meditación consiste en sentarme a respirar hondo mientras pienso en la enorme lista de tareas pendientes que no voy a realizar hoy.",
    "Qué gran detalle que nos den una tarjeta de felicitación impresa en papel reciclado para celebrar que logramos duplicar las ventas de la empresa.",
    "El asistente virtual de la web es sumamente útil, siempre responde con preguntas genéricas que no tienen nada que ver con mi consulta.",
    "Adoro cuando me dicen 'tenemos un ambiente de trabajo dinámico' y en realidad solo significa que cambian de opinión cada veinte minutos.",
    "Mi cuenta de ahorros está tan protegida que ni yo mismo puedo retirar dinero porque el saldo disponible es de cero con cincuenta centavos.",
    "Qué bien pensadas están las aplicaciones de citas, ideales para coleccionar saludos incómodos y conversaciones que mueren a los dos mensajes.",
    "Mi planta de menta ha decidido invadir toda la maceta de las flores vecinas, un gran ejemplo de expansionismo territorial botánico.",
    "Adoro cuando el repartidor de paquetes pone 'destinatario ausente' cuando llevo todo el día sentado junto a la puerta esperando el timbre.",
    "Qué gran avance el coche autónomo, ahora puedes experimentar el pánico de que un ordenador decida el rumbo de tu vida en una autopista transitada.",
    "Mi plan para combatir el estrés consiste en comer chocolate hasta que olvide el motivo de mi preocupación o se me acabe el chocolate.",
    "Qué suerte tener un trabajo donde puedes aprender a gestionar la frustración en tiempo real y sin tener que pagar un curso de psicología.",
    "Adoro cuando me dicen que sea creativo pero especifican que debo seguir el formato rígido del documento de mil novecientos ochenta y cinco.",
    "Mi gato prefiere dormir sobre el teclado caliente de mi laptop antes que en la cama acolchada de cincuenta euros que le compré ayer.",
    "Qué oportuno que el dentista me cobre el equivalente a mi salario de un mes por una consulta de diez minutos donde solo me dijo que tengo dientes.",
    "El curso de gestión del tiempo me enseñó que si planifico cada minuto de mi día de forma detallada, tendré aún más ansiedad cuando no cumpla nada.",
    "Adoro cuando el seguro del coche me cubre todo tipo de accidentes excepto aquellos que ocurren en la carretera o involucran a otros vehículos.",
    "Qué bien se siente vivir en la era de la información, donde puedes leer cien opiniones diferentes de expertos sobre por qué tu dolor de cabeza es mortal.",
    "Mi perro me mira con una cara de decepción tremenda cada vez que me ve comer pizza sin compartirle el borde de la masa.",
    "Qué gran idea obligarnos a hacer reuniones de inicio de semana a las ocho de la mañana para motivarnos a base de bostezos y café frío.",
    "Adoro cuando me dicen 'confía en el proceso' y el proceso consiste en dar vueltas en círculos hasta que el problema se resuelva solo por aburrimiento.",
    "Mi cuenta corriente se encuentra en un estado de minimalismo extremo, sin saldo que perturbe la paz mental de mi billetera vacía.",
    "Qué bien pensados están los menús interactivos de las pizzerías, ideales para pasar media hora decidiendo ingredientes y terminar pidiendo lo mismo.",
    "Mi planta de interior ha decidido secarse solo porque le dio una corriente de aire durante tres segundos, qué carácter tan caprichoso.",
    "Adoro cuando el técnico del internet dice que vendrá entre las ocho de la mañana y las ocho de la noche, ideal para pasar el día meditando junto a la puerta.",
    "Qué gran avance la medicina moderna, ahora puedes saber exactamente qué enfermedad tienes buscando en internet y asustándote a muerte en tres minutos.",
    "Mi perro ha decidido que el mejor momento para ladrarle a la pared vacía de la sala es a las tres de la madrugada, un excelente vigilante nocturno.",
    "Qué oportuno que se termine la batería de mis auriculares justo cuando un señor a mi lado en el autobús decide empezar a hablar de sus problemas estomacales.",
    "El manual de etiqueta corporativa me aconseja sonreír siempre, incluso cuando mi jefe me pide que rehaga el informe que me tomó tres días terminar.",
    "Adoro cuando me dicen que el dinero no compra la felicidad personas que viajan en yates privados y tienen cuentas bancarias en paraísos fiscales.",
    "Mi gato pasa el día entero persiguiendo una mota de polvo invisible en el aire, una agenda ocupada y de alta relevancia científica.",
    "Qué gran idea implementar el pago electrónico obligatorio en un estacionamiento que no tiene señal de datos móviles en el sótano.",
    "Adoro cuando me dicen 'aprende de tus errores' y luego me despiden por equivocarme en el formato del correo de presentación.",
    "Mi nevera tiene tantas botellas de salsa a medio terminar y tan poca comida real que parece una exposición de condimentos del siglo pasado.",
    "Qué bien se viaja en el metro de la ciudad, una experiencia inmersiva de calor humano y música variada en los altavoces de los pasajeros.",
    "Mi planta de aloe vera es la única que sobrevive en mi balcón, probablemente porque se alimenta del aire contaminado y de mi total negligencia.",
    "Adoro cuando me dicen 'trabajamos bajo presión' en la entrevista de trabajo, es un bonito sinónimo de 'aquí todos gritamos y corremos en círculos'.",
    "Qué gran detalle que la aerolínea me cobre por llevar una maleta de mano pequeña, seguro es para financiar la investigación aeroespacial de la empresa.",
    "El sistema de navegación gps de mi coche es excelente, siempre me sugiere rutas alternativas que involucran calles sin salida y zonas en construcción.",
    "Adoro cuando me dicen 'toma la iniciativa' y luego me cancelan el proyecto porque no estaba alineado con el pensamiento del subdirector de área.",
    "Mi gato prefiere meterse en la caja de cartón vacía del paquete antes que usar el rascador de tres pisos que me costó una fortuna.",
    "Qué oportuno que el fontanero me diga que la fuga de agua es sencilla pero requiere cambiar toda la instalación de cañerías de la casa.",
    "El taller de productividad personal me enseñó a crear listas de tareas infinitas para sentirme culpable por no terminar ninguna al final del día.",
    "Adoro cuando la compañía de seguros me pide fotos del daño del coche tomadas desde catorce ángulos diferentes y con luz solar directa.",
    "Qué bien pensada está la burocracia estatal: pasas tres horas en una fila para que te digan que el documento que traes debe llevar un sello de otra oficina.",
    "Mi perro pasa la tarde persiguiéndose la cola en círculos sobre la alfombra de la sala, un pasatiempo de alta complejidad intelectual.",
    "Qué suerte vivir en una época donde puedes recibir quinientas notificaciones al día de aplicaciones de comida que te recuerdan que tienes hambre.",
    "Adoro cuando me dicen 'este es un proyecto retador' y resulta que solo significa que no tenemos presupuesto ni personal para realizarlo.",
    "Mi cuenta de ahorros está tan vacía que si me roban la tarjeta de débito, el ladrón probablemente termine depositándome dinero por lástima.",
    "Qué bien diseñadas están las bolsas de patatas fritas: pagas por un 80% de aire de primera calidad y un 20% de patatas de regalo.",
    "Mi planta de interior ha decidido marchitarse solo porque la cambié de ventana a una con tres centímetros más de sombra, una diva total.",
    "Adoro cuando el cartero deja una nota de 'intento de entrega fallido' cuando he pasado todo el día sentado junto a la ventana esperándolo.",
    "Qué gran avance de la tecnología: ahora puedes programar un robot para que barra tu casa mientras tú pasas el día pegado a la pantalla del móvil.",
    "Mi perro considera que el mejor lugar para lamerse las patas con un sonido molesto es justo debajo de mi silla durante mis videollamadas de trabajo.",
    "Qué oportuno que el supermercado decida subir el precio de los huevos justo la semana en que decidí iniciar una dieta basada en proteínas.",
    "El manual de autoayuda me dice que debo amarme a mí mismo, lo cual es complicado cuando me veo en el espejo con los pelos despeinados de la mañana.",
    "Adoro cuando me dicen que sea espontáneo y flexible pero me exigen que reporte mis actividades diarias en bloques de quince minutos.",
    "Mi gato pasa la noche entera maullándole a una puerta cerrada que da a una habitación vacía, un filósofo del existencialismo felino.",
    "Qué gran idea implementar un sistema de turnos digital en un consultorio donde el médico atiende por orden de llegada de sus conocidos.",
    "Adoro cuando me dan consejos de vida saludable personas que tienen un cocinero personal y un entrenador de planta en sus mansiones.",
    "Mi nevera tiene tantas sobras de comida de la semana pasada en envases de plástico que ya parece un laboratorio de biología experimental.",
    "Qué bien se siente viajar en bus en la ciudad, sobre todo cuando el chofer decide frenar de golpe para probar los reflejos de los pasajeros.",
    "Mi planta de romero se secó por completo, demostrando que ni siquiera las hierbas aromáticas más resistentes pueden sobrevivir a mi cuidado.",
    "Adoro cuando me dicen 'tenemos grandes beneficios laborales' y el beneficio estrella es tener café gratis en una cafetera que nunca limpian.",
    "Qué gran detalle de la empresa proveedora de luz: me aumentaron la tarifa mensual para motivarme a usar velas y ahorrar energía en el hogar.",
    "El asistente virtual del banco es sumamente inteligente, siempre me bloquea la cuenta cuando intento hacer una transferencia de emergencia.",
    "Adoro cuando la página web del gobierno me pide que acceda con un navegador de internet específico de mil novecientos noventa y ocho.",
    "Mi gato prefiere dormir sobre mi cara a las cuatro de la mañana antes que usar la almohada especial para mascotas que le compré por internet.",
    "Qué oportuno que se descomponga el aire acondicionado de la oficina justo el día en que la temperatura exterior alcanza los cuarenta grados.",
    "El curso de oratoria me enseñó que si hablo con tono firme y pausado, la gente no notará que no tengo idea del tema del que estoy hablando.",
    "Adoro cuando me dicen que la felicidad está en las pequeñas cosas personas que poseen colecciones de coches deportivos y viajan en jet privado.",
    "Mi perro pasa horas intentando atrapar una mosca en el salón, demostrando una constancia digna de un deportista olímpico.",
    "Qué bien pensado el sistema de cobro automático del peaje, siempre decide no leer mi tarjeta y me obliga a retroceder ante cien coches molestos.",
    "Adoro cuando me dicen 'aprende de tus errores' y luego me cancelan la beca de estudios por haber fallado en una sola pregunta del examen.",
    "Mi nevera está tan desprovista de alimentos que el hielo del congelador ya comenzó a tomar un sabor a cebolla y decepción existencial.",
    "Qué bien se viaja en el metro de la ciudad en verano, una sauna móvil gratuita que te permite socializar de cerca con extraños sudorosos.",
    "Mi planta de cactus ha decidido secarse por completo, rompiendo todas las leyes de la botánica y de la supervivencia en climas áridos.",
    "Adoro cuando en la entrevista de trabajo me dicen que valoran la honestidad y luego se ofenden cuando les pregunto por el salario neto mensual.",
    "Qué gran avance de las redes sociales: ahora puedes enterarte de la vida perfecta de tus excompañeros de escuela mientras comes fideos instantáneos en tu cama.",
    "Mi gato considera que el mejor momento para correr a máxima velocidad por toda la casa tirando objetos es a las dos de la madrugada.",
    "Qué oportuno que el calentador de agua decida dejar de funcionar justo cuando tengo jabón en los ojos y me estoy bañando con agua helada.",
    "El manual de etiqueta laboral me sugiere ser empático con mis jefes, sobre todo cuando me piden que trabaje horas extras el día de mi cumpleaños.",
    "Adoro cuando me dicen 'el dinero no lo es todo' personas que nunca han tenido que elegir entre pagar el alquiler o comprar la comida de la semana.",
    "Mi perro pasa la mañana persiguiendo el reflejo de la luz del sol en la pared, un trabajo de alta relevancia física y astronómica.",
    "Qué gran idea poner el botón de cancelar suscripción en un menú oculto detrás de tres páginas de encuestas de satisfacción obligatorias.",
    "Adoro cuando me dicen 'confía en el equipo' y el equipo consiste en mí haciendo todo el trabajo mientras los demás aprueban mis ideas en las reuniones.",
    "Mi cuenta bancaria se encuentra en un estado de sobriedad absoluta, libre de cualquier cifra que pueda generar tentaciones de consumo.",
    "Qué bien pensados están los empaques de galletas: pagas por un envoltorio brillante enorme que contiene cuatro galletas rotas en el fondo.",
    "Mi planta de interior se marchitó por completo porque olvidé abrir la persiana durante una mañana, una susceptibilidad digna de estudio científico.",
    "Adoro cuando el mensajero pone que el paquete fue entregado al vecino y el vecino resulta ser un terreno baldío al final de la calle.",
    "Qué gran avance de la domótica: ahora puedes pasar media hora intentando encender la luz de la sala mediante una aplicación que no se conecta al wifi.",
    "Mi perro ha decidido que su juguete favorito es una botella de plástico vacía que hace un ruido ensordecedor cada vez que la muerde en la noche.",
    "Qué oportuno que la aerolínea decida cancelar mi vuelo de conexión justo el día en que tengo la entrevista final para el trabajo de mis sueños.",
    "El manual de autoayuda me aconseja sonreír al espejo cada mañana, pero el espejo me devuelve una mirada de 'por favor vuelve a acostarte'.",
    "Adoro cuando me piden que trabaje con autonomía pero debo pedir autorización para comprar un paquete de folios de papel para la impresora.",
    "Mi gato pasa la tarde entera maullándole a la pared del pasillo, como si estuviera conversando con fantasmas del siglo pasado.",
    "Qué bien pensado el sistema de estacionamiento de la ciudad: pagas una fortuna por dejar tu coche en una calle llena de baches sin vigilancia.",
    "Adoro cuando me dan consejos de alimentación saludable personas que tienen un chef privado que les cocina verduras orgánicas frescas todos los días.",
    "Mi nevera tiene tantos frascos de mermelada casi vacíos que ya parece una colección de muestras biológicas de una expedición científica.",
    "Qué bien se siente viajar en autobús en la hora pico, una experiencia ideal para conocer de cerca los diferentes tipos de desodorante de la población.",
    "Mi planta de menta ha decidido secarse misteriosamente a pesar de estar en una maceta con tierra de primera calidad, qué carácter tan rebelde.",
    "Adoro cuando me dicen que la empresa tiene un 'ambiente dinámico y flexible' y resulta que solo significa que no tienen horarios de salida.",
    "Qué gran detalle de la compañía de gas: me aumentaron la factura mensual para motivarme a tomar baños fríos y fortalecer mi sistema inmunológico.",
    "El chatbot del servicio técnico es sumamente eficiente, siempre me deriva con otro robot que me hace las mismas preguntas desde el principio.",
    "Adoro cuando la página de la universidad me pide que suba mi foto en formato JPG de exactamente trescientos por trescientos píxeles y cincuenta kilobytes.",
    "Mi gato prefiere dormir dentro de una bolsa de plástico ruidosa antes que usar la cama térmica de lujo que le compré por su cumpleaños.",
    "Qué oportuno que se rompa la tubería del baño justo el día en que tengo planeado salir de viaje de vacaciones por una semana.",
    "El curso de liderazgo moderno me enseñó que si uso palabras en inglés como 'target' y 'deadline', mis propuestas sonarán el doble de costosas.",
    "Adoro cuando me dicen 'el dinero no hace la felicidad' personas que tienen cuentas millonarias en Suiza y viajan en helicóptero privado.",
    "Mi perro pasa horas intentando morder el chorro de agua de la manguera del jardín, una actividad de alta complejidad física.",
    "Qué bien pensado el lector de tarjetas del metro, siempre decide dar error de lectura cuando tengo a cincuenta personas apuradas detrás de mí.",
    "Adoro cuando me dicen 'aprende de tus fallos' y luego me cancelan la beca de investigación por haber entregado el informe con un día de retraso.",
    "Mi nevera está tan vacía que la única verdura que contiene es un limón seco que ha estado ahí desde el inicio de la administración actual."
]

# Unificar y procesar
selected_features = [
    'MeanWordLen', 'LexicalDiversity', 'MeanSentenceLen', 'StdevSentenceLen', 'DocumentLen',
    'WordsPerText', 'SentencesPerText', 'num_words', 'num_chars', 'irony_score',
    'prop_NOUN', 'prop_VERB', 'prop_ADJ', 'rhetorical_questions', 'avg_depth',
    'Flesch Score', 'Lexical Entropy', 'Syntactic Repetition', 'Unusual Word Frequency'
]

def agregar_y_procesar_500():
    ruta_original = os.path.join("documentos_origen", "Titulacion1", "DatasetsFinales", "df_train2_featselect2.jsonl")
    
    if not os.path.exists(ruta_original):
        print(f"[ERROR] No se encontro el dataset original en {ruta_original}")
        return
        
    print(f"[LOAD] Cargando dataset original ({ruta_original})...")
    df_original = pd.read_json(ruta_original, orient='records', lines=True)
    print(f"[INFO] Dataset cargado. Contiene {len(df_original)} registros.")

    processor = TextProcessor()
    
    # Armar los 500 nuevos ejemplos (250 serios, 250 satíricos)
    ejemplos_nuevos_raw = []
    
    # Agregar los 250 serios
    for idx, text in enumerate(ejemplos_serios[:250]):
        ejemplos_nuevos_raw.append({"text": text, "label": 0, "id_prefix": "refuerzo500_serio"})
        
    # Agregar los 250 satíricos
    for idx, text in enumerate(ejemplos_satiricos[:250]):
        ejemplos_nuevos_raw.append({"text": text, "label": 1, "id_prefix": "refuerzo500_satirico"})
        
    nuevos_registros = []
    
    print("\n[INFO] Procesando caracteristicas linguisticas de los 500 ejemplos de refuerzo cotidiano...")
    for idx, ejemplo in enumerate(ejemplos_nuevos_raw):
        text = ejemplo["text"]
        label = ejemplo["label"]
        prefix = ejemplo["id_prefix"]
        
        # Generar texto procesado
        processed_text = processor.preprocess_text(text)
        # Extraer las 19 características manuales
        features_dict = processor.calculate_features(text)
        
        # Armar el registro equivalente a las columnas del JSONL
        registro = {
            "id": f"{prefix}_{idx}",
            "transcription": text,
            "transcription_processed": processed_text,
            "label": label
        }
        
        # Añadir las características calculadas al registro en el orden correcto
        for feat in selected_features:
            registro[feat] = features_dict.get(feat, 0)
            
        nuevos_registros.append(registro)
        
    df_nuevos = pd.DataFrame(nuevos_registros)
    print(f"[INFO] Procesados {len(df_nuevos)} nuevos registros de refuerzo cotidiano.")
    
    # Combinar datasets (Original de 6100 + Nuevos 500 = 6600)
    df_expandido = pd.concat([df_original, df_nuevos], ignore_index=True)
    
    # Guardar el dataset expandido
    df_expandido.to_json(ruta_original, orient='records', lines=True)
    print(f"[SAVE] Dataset expandido guardado en {ruta_original} (Total: {len(df_expandido)} registros).")
    
    # 3. Re-entrenar y guardar los serializadores locales (TF-IDF y MinMaxScaler)
    print("\n[INFO] Re-ajustando Vectorizador TF-IDF y Escalador MinMaxScaler con el corpus expandido...")
    
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_features = vectorizer.fit_transform(df_expandido['transcription_processed'].fillna("")).toarray()
    
    manual_features = df_expandido[selected_features].values
    combined_features = np.concatenate([tfidf_features, manual_features], axis=1)
    
    scaler = MinMaxScaler()
    scaler.fit(combined_features)
    
    # Guardar en static local
    static_dir = os.path.join("satire_detector_api", "static")
    os.makedirs(static_dir, exist_ok=True)
    
    joblib.dump(vectorizer, os.path.join(static_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(static_dir, "minmax_scaler.pkl"))
    print(f"[INFO] tfidf_vectorizer.pkl y minmax_scaler.pkl actualizados con exito en {static_dir}!")
    
    print("\n" + "="*60)
    print("PROCESAMIENTO LOCAL COMPLETADO CON ÉXITO")
    print("="*60)

if __name__ == "__main__":
    agregar_y_procesar_500()
