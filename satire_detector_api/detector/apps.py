from django.apps import AppConfig
import os

class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    def ready(self):
        # Evitar doble carga en el servidor de desarrollo de Django (con auto-reload)
        if os.environ.get('RUN_MAIN') != 'true' and os.environ.get('DJANGO_SETTINGS_MODULE') == 'satire_detector_api.settings':
            # Si estamos usando runserver y no es el subproceso activo, no cargamos aún
            return
            
        print("[DetectorConfig] Pre-cargando modelos en la inicialización del backend...")
        try:
            from detector.utils.text_processor import TextProcessor
            # Forzar inicialización de la instancia única (singleton)
            processor = TextProcessor()
            print("[DetectorConfig] Modelos cargados con éxito en memoria.")
        except Exception as e:
            print(f"[DetectorConfig] Error al pre-cargar los modelos: {e}")

