import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-metodologia',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './metodologia.component.html',
  styleUrl: './metodologia.component.css'
})
export class MetodologiaComponent {
  /**
   * Define la pestaña activa actualmente.
   */
  seccionActiva: string = "fases";

  /**
   * Actualiza la pestaña activa para alternar entre las diferentes secciones metodológicas.
   * @param seccion Identificador de la sección seleccionada.
   */
  seleccionarSeccion(seccion: string) {
    this.seccionActiva = seccion;
  }
}
