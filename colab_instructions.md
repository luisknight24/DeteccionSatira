# Guía de Inicialización del Backend de Detección de Sátira en Google Colab

Esta guía explica paso a paso cómo iniciar y exponer el servidor backend Django en Google Colab utilizando el cuaderno interactivo [iniciar_servidor_colab.ipynb](file:///c:/Users/Luis/Desktop/Antigravity/DeteccionSatira/iniciar_servidor_colab.ipynb) y conectarlo con tu Frontend local.

---

## Prerrequisitos
1. El proyecto actual debe estar subido a GitHub (esto ya se realizó en el commit anterior).
2. Tener tu navegador web abierto.

---

## Paso 1: Subir el Cuaderno a Google Colab
1. Ve a [Google Colab](https://colab.research.google.com/).
2. Haz clic en la pestaña **Subir** (Upload).
3. Selecciona el archivo [iniciar_servidor_colab.ipynb](file:///c:/Users/Luis/Desktop/Antigravity/DeteccionSatira/iniciar_servidor_colab.ipynb) desde la carpeta raíz de tu proyecto.

---

## Paso 2: Ejecutar las Celdas en Google Colab

### Celda 1: Clonar el Repositorio de GitHub
Ejecuta la primera celda de código. Esta celda clonará todo tu proyecto (incluyendo los pesos del modelo y los datasets que agregaste a `documentos_origen/`) dentro de la sesión temporal de Google Colab.

### Celda 2: Instalar Dependencias
Ejecuta la segunda celda de código. Esta celda:
* Instalará las librerías necesarias del servidor.
* **Instalará la versión compatible de transformers (`4.44.2`)** para asegurar que el modelo se deserialice correctamente sin fallos de importación.
* Descargará los diccionarios de lematización en español de spaCy y los recursos de NLTK.

### Celda 3: Copiar Archivos del Modelo
Antes de ejecutar esta celda, asegúrate de subir los archivos del modelo real a tu Google Drive:
1. Crea una carpeta en tu Google Drive llamada `Titulacion` y, dentro de ella, una subcarpeta llamada `Modelos` (de modo que la ruta en Drive sea: `Mi unidad/Titulacion/Modelos/`).
2. Sube a esa carpeta de Drive los siguientes archivos:
   * `best_model_spanish_loss.pt`
   * `tfidf_vectorizer.pkl`
   * `minmax_scaler.pkl`
   * El directorio `tokenizer_files/` completo.
3. Ejecuta la celda. El cuaderno copiará estos archivos automáticamente a la carpeta `static/` de Django en Colab. Si no los tienes en Drive, el script intentará cargarlos alternativamente desde los archivos del repositorio si están disponibles localmente.

### Celda 4: Iniciar el Backend y Crear el Túnel Público
Ejecuta la cuarta celda.
1. La celda iniciará Django en segundo plano y esperará a que cargue el modelo en memoria.
2. Imprimirá un texto que dice **`🔑 COPIA ESTA IP PÚBLICA PARA AUTORIZAR LOCALTUNNEL:`** seguido de una dirección IP (por ejemplo: `34.125.12.87`). **Copia esa dirección IP**.
3. Iniciará `localtunnel` y mostrará un enlace web del túnel público (por ejemplo: `https://slimy-wolves-yell.loca.lt`). **Haz clic en el enlace**.
4. En la pestaña que se abrirá en tu navegador, te solicitará una contraseña o dirección IP. **Pega la IP que copiaste en el paso anterior** y presiona **Submit**.
5. ¡Listo! Verás la pantalla de inicio de la API de Django en la nube. Copia esa URL completa (la que termina en `.loca.lt/`).

---

## Paso 3: Conectar el Frontend Angular local con Colab
Una vez que obtengas la URL de localtunnel (ej: `https://xxxx.loca.lt`):
1. Inicia tu frontend localmente en tu terminal: `npm run start` (o `ng serve`).
2. Abre la aplicación en tu navegador (`http://localhost:4200`).
3. En la barra de navegación superior, haz clic en el icono de **Configuración ⚙️** (se agregará a continuación).
4. Pega la URL del túnel de Colab en el campo del input de la API y haz clic en **Guardar**.
5. ¡El detector comenzará a resolver las predicciones de forma instantánea usando el modelo BETO real alojado en Google Colab!
