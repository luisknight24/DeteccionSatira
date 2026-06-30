import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';
import { map } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export interface RespuestaSatira {
  prediccion: string;
  probabilidad: number;
  metricas?: {
    irony_score: number;
    LexicalDiversity: number;
    'Flesch Score': number;
    'Unusual Word Frequency': number;
    prop_NOUN: number;
    prop_VERB: number;
    prop_ADJ: number;
  };
}

// Interfaz para la respuesta cruda de la API del backend en inglés
interface RespuestaApiBackend {
  prediction: string;
  probability: number;
  metrics?: {
    irony_score: number;
    LexicalDiversity: number;
    'Flesch Score': number;
    'Unusual Word Frequency': number;
    prop_NOUN: number;
    prop_VERB: number;
    prop_ADJ: number;
  };
}

@Injectable({
  providedIn: 'root'
})
export class ServicioSatira {
  // Clave para almacenar la URL del backend en localStorage
  private readonly CLAVE_ALMACENAMIENTO = 'url_backend_satira';
  
  // URL predeterminada del backend (cargada según el entorno)
  private readonly URL_PREDETERMINADA = environment.apiUrl;

  // Canal reactivo para alertar sobre cambios en la configuración del API
  private urlBaseSubject = new BehaviorSubject<string>(this.obtenerUrlBase());
  urlBase$ = this.urlBaseSubject.asObservable();

  constructor(private http: HttpClient) { }

  // Obtiene la URL base activa persistida o retorna el valor predeterminado
  obtenerUrlBase(): string {
    let urlGuardada = localStorage.getItem(this.CLAVE_ALMACENAMIENTO);
    if (urlGuardada) {
      urlGuardada = urlGuardada.trim();
      // Auto-corregir si falta el segmento /api al final de la URL personalizada
      if (!urlGuardada.endsWith('/api') && !urlGuardada.endsWith('/api/')) {
        if (urlGuardada.endsWith('/')) {
          urlGuardada = urlGuardada.slice(0, -1);
        }
        urlGuardada = `${urlGuardada}/api`;
      }
      return urlGuardada.endsWith('/') ? urlGuardada : `${urlGuardada}/`;
    }
    return this.URL_PREDETERMINADA;
  }

  // Actualiza la URL del API de forma definitiva en memoria y almacenamiento local
  actualizarUrlBase(nuevaUrl: string): void {
    let urlFormateada = nuevaUrl.trim();
    if (urlFormateada) {
      if (!urlFormateada.endsWith('/')) {
        urlFormateada = `${urlFormateada}/`;
      }
      localStorage.setItem(this.CLAVE_ALMACENAMIENTO, urlFormateada);
      this.urlBaseSubject.next(urlFormateada);
    }
  }

  // Resetea la URL del API a la opción predeterminada
  restablecerUrlBase(): void {
    localStorage.removeItem(this.CLAVE_ALMACENAMIENTO);
    this.urlBaseSubject.next(this.URL_PREDETERMINADA);
  }

  // Envía el texto plano al backend de Django expuesto para la extracción de características y clasificación
  detectarSatira(texto: string): Observable<RespuestaSatira> {
    const urlCompleta = `${this.obtenerUrlBase()}satire-detection/`;
    // Añadimos cabecera para omitir la página de advertencia de localtunnel programáticamente
    const headers = new HttpHeaders({
      'bypass-tunnel-reminder': 'true'
    });
    return this.http.post<RespuestaApiBackend>(urlCompleta, { text: texto }, { headers }).pipe(
      map(respuesta => ({
        prediccion: respuesta.prediction,
        probabilidad: respuesta.probability,
        metricas: respuesta.metrics
      }))
    );
  }
}
