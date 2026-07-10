import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject, of } from 'rxjs';
import { map, catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export interface RespuestaSatira {
  prediccion: string;
  probabilidad: number;
  isMock?: boolean;
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

  // Simulación local en Javascript del detector de sátira en caso de servidor offline
  private simularPrediccionLocal(texto: string): RespuestaSatira {
    const palabras = texto.trim().toLowerCase().split(/\s+/);
    const conteoPalabras = palabras.length;
    const exclamaciones = (texto.match(/!/g) || []).length;
    
    const terminosSatiricos = [
      'obviamente', 'claro', 'supuestamente', 'genial', 'increíble', 'absurdo', 
      'político', 'candidato', 'elecciones', 'gobierno', 'promete', 'promesas',
      'impuestos', 'reforma', 'bache', 'carretera', 'diputado', 'presidente',
      'fútbol', 'madrid', 'barcelona', 'sarcasmo', 'ironía', 'maravilloso', 
      'excelente', 'milagro', 'histórico', 'patrimonio'
    ];
    
    let coincidencias = 0;
    palabras.forEach(palabra => {
      const palabraLimpia = palabra.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()¿?]/g,"");
      if (terminosSatiricos.includes(palabraLimpia)) {
        coincidencias++;
      }
    });

    let scoreVal = 0.15; // Probabilidad base de 15%
    scoreVal += coincidencias * 0.12;
    scoreVal += exclamaciones * 0.08;
    if (conteoPalabras > 40) scoreVal += 0.10;
    if (conteoPalabras < 10) scoreVal -= 0.05;
    scoreVal = Math.max(0.01, Math.min(0.99, scoreVal));
    
    const prediccion = scoreVal >= 0.55 ? 'ES SÁTIRA' : 'NO ES SÁTIRA';
    const LexicalDiversity = Math.min(1, Math.max(0.4, 1 - (conteoPalabras - new Set(palabras).size) / (conteoPalabras || 1)));
    const irony_score = Math.min(1, Math.max(0, (coincidencias * 0.15) + (exclamaciones * 0.1)));
    const oraciones = (texto.match(/[.!?]+/g) || []).length || 1;
    const promedioPalabrasPorOracion = conteoPalabras / oraciones;
    const FleschScore = Math.max(0, Math.min(100, 206.84 - (1.02 * promedioPalabrasPorOracion) - (60 * LexicalDiversity)));

    const prop_NOUN = Math.min(0.5, Math.max(0.15, 0.25 + (Math.sin(conteoPalabras) * 0.05)));
    const prop_VERB = Math.min(0.4, Math.max(0.1, 0.18 + (Math.cos(conteoPalabras) * 0.04)));
    const prop_ADJ = Math.min(0.3, Math.max(0.02, 0.1 + (exclamaciones * 0.05)));
    
    return {
      prediccion,
      probabilidad: scoreVal,
      isMock: true,
      metricas: {
        irony_score: Number(irony_score.toFixed(2)),
        LexicalDiversity: Number(LexicalDiversity.toFixed(2)),
        'Flesch Score': Number(FleschScore.toFixed(1)),
        'Unusual Word Frequency': Number((prop_ADJ * 0.3).toFixed(2)),
        prop_NOUN: Number(prop_NOUN.toFixed(2)),
        prop_VERB: Number(prop_VERB.toFixed(2)),
        prop_ADJ: Number(prop_ADJ.toFixed(2))
      }
    };
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
        metricas: respuesta.metrics,
        isMock: false
      })),
      catchError(error => {
        console.warn('El servidor backend de Django no está disponible. Iniciando demostración simulada local:', error);
        return of(this.simularPrediccionLocal(texto));
      })
    );
  }
}
