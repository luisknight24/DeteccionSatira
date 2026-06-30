import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export interface EntradaHistorial {
  id: string;
  texto: string;
  prediccion: string;
  probabilidad: number;
  fecha: string;
}

@Injectable({
  providedIn: 'root'
})
export class ServicioHistorial {
  private claveAlmacenamiento = 'satire_detector_history';
  
  // Flujo para transferir una consulta previa del historial al formulario activo
  public entradaSeleccionada$ = new BehaviorSubject<EntradaHistorial | null>(null);

  // Obtiene los registros almacenados localmente en el navegador
  obtenerHistorial(): EntradaHistorial[] {
    const historial = localStorage.getItem(this.claveAlmacenamiento);
    return historial ? JSON.parse(historial) : [];
  }

  // Agrega una nueva consulta al inicio del listado de historial local
  agregarEntrada(texto: string, prediccion: string, probabilidad: number): void {
    const historial = this.obtenerHistorial();
    const nuevaEntrada: EntradaHistorial = {
      id: Math.random().toString(36).substring(2, 9),
      texto,
      prediccion,
      probabilidad,
      fecha: new Date().toLocaleString('es-ES')
    };
    historial.unshift(nuevaEntrada);
    
    // Restringe el almacenamiento local a las últimas 20 consultas
    if (historial.length > 20) {
      historial.pop();
    }
    localStorage.setItem(this.claveAlmacenamiento, JSON.stringify(historial));
  }

  // Borra la persistencia de consultas anteriores del localStorage
  limpiarHistorial(): void {
    localStorage.removeItem(this.claveAlmacenamiento);
  }
}
