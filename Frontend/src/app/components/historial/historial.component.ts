import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { ServicioHistorial, EntradaHistorial } from '../../services/servicio-historial.service';

@Component({
  selector: 'app-historial',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './historial.component.html',
  styleUrl: './historial.component.css'
})
export class HistorialComponent implements OnInit {
  listaHistorial: EntradaHistorial[] = [];

  constructor(
    private servicioHistorial: ServicioHistorial,
    private enrutador: Router
  ) {}

  ngOnInit(): void {
    this.cargarHistorial();
  }

  // Carga las consultas persistidas localmente
  cargarHistorial(): void {
    this.listaHistorial = this.servicioHistorial.obtenerHistorial();
  }

  // Remueve de forma definitiva todas las consultas previas
  limpiarHistorial(): void {
    this.servicioHistorial.limpiarHistorial();
    this.cargarHistorial();
  }

  // Envía la consulta seleccionada al panel del detector mediante el canal compartido
  seleccionarEntradaHistorial(entrada: EntradaHistorial): void {
    this.servicioHistorial.entradaSeleccionada$.next(entrada);
    this.enrutador.navigate(['/detector']);
  }
}
