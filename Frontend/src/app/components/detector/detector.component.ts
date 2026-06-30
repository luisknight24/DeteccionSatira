import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { ServicioSatira, RespuestaSatira } from '../../services/servicio-satira.service';
import { ServicioHistorial } from '../../services/servicio-historial.service';

@Component({
  selector: 'app-detector',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './detector.component.html',
  styleUrl: './detector.component.css'
})
export class DetectorComponent implements OnInit, OnDestroy {
  textoIngresado: string = '';
  cargando: boolean = false;
  mensajeError: string = '';
  
  resultado: RespuestaSatira | null = null;
  textoAnalizado: string = '';

  private subscripcionHistorial?: Subscription;

  constructor(
    private servicioSatira: ServicioSatira,
    private servicioHistorial: ServicioHistorial
  ) {}

  ngOnInit(): void {
    // Escucha solicitudes de restauración provenientes del historial
    this.subscripcionHistorial = this.servicioHistorial.entradaSeleccionada$.subscribe({
      next: (entrada) => {
        if (entrada) {
          this.textoIngresado = entrada.texto;
          this.resultado = {
            prediccion: entrada.prediccion,
            probabilidad: entrada.probabilidad
          };
          this.textoAnalizado = entrada.texto;
          // Libera el flujo una vez consumido el estado
          this.servicioHistorial.entradaSeleccionada$.next(null);
        }
      }
    });
  }

  ngOnDestroy(): void {
    if (this.subscripcionHistorial) {
      this.subscripcionHistorial.unsubscribe();
    }
  }

  // Getters para estadísticas en tiempo real del texto
  get conteoCaracteres(): number {
    return this.textoIngresado ? this.textoIngresado.length : 0;
  }

  get conteoPalabras(): number {
    if (!this.textoIngresado.trim()) return 0;
    return this.textoIngresado.trim().split(/\s+/).length;
  }

  get tiempoLectura(): number {
    const palabras = this.conteoPalabras;
    return Math.ceil(palabras / 200);
  }

  get conteoExclamaciones(): number {
    if (!this.textoIngresado) return 0;
    return (this.textoIngresado.match(/!/g) || []).length;
  }

  // Envía el texto a clasificar por el modelo neuronal
  analizarTexto(): void {
    if (!this.textoIngresado.trim()) return;
    
    this.cargando = true;
    this.mensajeError = '';
    this.resultado = null;
    
    this.servicioSatira.detectarSatira(this.textoIngresado).subscribe({
      next: (res) => {
        this.resultado = res;
        this.textoAnalizado = this.textoIngresado;
        this.servicioHistorial.agregarEntrada(this.textoIngresado, res.prediccion, res.probabilidad);
        this.cargando = false;
      },
      error: (err) => {
        console.error('Error al invocar API:', err);
        this.mensajeError = 'No se pudo conectar con el servidor backend de Django. Asegúrate de levantar el servidor ejecutando: python manage.py runserver 8001';
        this.cargando = false;
      }
    });
  }
}
