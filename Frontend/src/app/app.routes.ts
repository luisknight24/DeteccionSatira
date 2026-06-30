import { Routes } from '@angular/router';
import { DetectorComponent } from './components/detector/detector.component';
import { HistorialComponent } from './components/historial/historial.component';
import { MetodologiaComponent } from './components/metodologia/metodologia.component';

export const routes: Routes = [
  // Ruta por defecto para cargar el panel de detección sintáctico-semántica
  { path: 'detector', component: DetectorComponent },
  
  // Ruta para consultar las peticiones almacenadas en la sesión activa
  { path: 'historial', component: HistorialComponent },
  
  // Ruta informativa para la explicación del modelo híbrido
  { path: 'metodologia', component: MetodologiaComponent },
  
  // Redirecciones por defecto y comodín ante rutas inexistentes
  { path: '', redirectTo: 'detector', pathMatch: 'full' },
  { path: '**', redirectTo: 'detector' }
];
