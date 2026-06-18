from waitress import serve
from satire_detector_api.wsgi import application  # Asegúrate que este nombre es correcto

if __name__ == "__main__":
    serve(application, host="0.0.0.0", port=8000, threads=4)
