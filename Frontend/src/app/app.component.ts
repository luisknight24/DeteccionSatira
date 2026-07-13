import { Component, OnInit } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ServicioSatira } from './services/servicio-satira.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive, CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
  mostrarPanelConfiguracion: boolean = false;
  urlServidorInput: string = '';

  constructor(private servicioSatira: ServicioSatira) { }

  ngOnInit(): void {
    this.cargarUrlActual();
  }

  // Carga la URL base configurada actualmente
  cargarUrlActual(): void {
    this.urlServidorInput = this.servicioSatira.obtenerUrlBase();
  }

  // Abre o cierra el panel de configuración del API
  alternarPanelConfiguracion(): void {
    this.mostrarPanelConfiguracion = !this.mostrarPanelConfiguracion;
    if (this.mostrarPanelConfiguracion) {
      this.cargarUrlActual();
    }
  }

  // Persiste la nueva URL y cierra el panel desplegable
  guardarConfiguracionUrl(): void {
    this.servicioSatira.actualizarUrlBase(this.urlServidorInput);
    this.mostrarPanelConfiguracion = false;
  }

  // Restablece la URL a la predeterminada del backend local
  restablecerUrlPredeterminada(): void {
    this.servicioSatira.restablecerUrlBase();
    this.cargarUrlActual();
    this.mostrarPanelConfiguracion = false;
  }
}
