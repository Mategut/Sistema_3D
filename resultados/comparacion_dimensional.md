# Comparación dimensional exploratoria

La comparación contrasta las mallas publicadas con las referencias aproximadas tomadas con regla. La geometría se conserva y se aplica el mismo método a cada tipo de objeto, sin ajustar los parámetros a sus dimensiones reales. Las discrepancias describen las diferencias obtenidas con ese procedimiento; no certifican la exactitud del sistema.

## Método y alcance

- Altura de cubo y cilindro: extensión sobre Y, eje vertical del sistema calibrado. Requiere objeto apoyado y alineado con el montaje; no equivale a ajustar un plano de base independiente. El JSON incluye también la extensión entre percentiles 0.5–99.5 como diagnóstico de extremos, sin sustituir la medida principal.
- Cubo: rectángulo envolvente de área mínima en XZ, corrigiendo el giro horizontal. Lado menor se compara con fondo y mayor con ancho; esa asignación por orden no identifica físicamente las caras. Inclinación y puntos extremos pueden aumentar las dimensiones.
- Cilindro: ajuste circular robusto en XZ sobre los vértices situados entre el 20 % y el 80 % de la altura. Se guarda el residual radial P95. El ajuste supone eje aproximadamente paralelo a Y y la densidad de vértices influye en la ponderación.
- Pirámide triangular: plano por SVD en la banda extrema de 2 mm con mayor área horizontal; esquinas por envolvente convexa simplificada al 4 % del perímetro; punta promediada en la banda extrema opuesta. Se guardan los puntos, los tres lados, aristas y alturas de cara, y se comparan sus medias con las referencias aproximadas. No se supone regularidad ni correspondencia individual de caras. Redondeo y banda de punta pueden subestimar medidas. La altura perpendicular al plano y la extensión Y son diagnósticos sin referencia física; los 85 mm corresponden a altura de cara. La fotografía solo identifica segmentos, no aporta medidas por píxeles.
- Diferencia = reconstruido − referencia. Porcentaje = 100 × diferencia / referencia. La desviación estándar muestral de las tres campañas describe variación de resultados, no incertidumbre de la regla ni independencia experimental acreditada.

## Mallas finales

| Campaña | Magnitud | Regla (mm) | Reconstrucción (mm) | Diferencia (mm) | Diferencia (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| cilindro1 | diametro | 43.0 | 44.69 | +1.69 | +3.94 |
| cilindro1 | altura | 97.0 | 99.18 | +2.18 | +2.24 |
| cilindro2 | diametro | 43.0 | 45.62 | +2.62 | +6.09 |
| cilindro2 | altura | 97.0 | 97.96 | +0.96 | +0.99 |
| cilindro3 | diametro | 43.0 | 44.66 | +1.66 | +3.87 |
| cilindro3 | altura | 97.0 | 99.02 | +2.02 | +2.08 |
| cubo1 | fondo | 94.0 | 98.42 | +4.42 | +4.70 |
| cubo1 | ancho | 95.0 | 98.91 | +3.91 | +4.11 |
| cubo1 | alto | 95.0 | 95.46 | +0.46 | +0.49 |
| cubo2 | fondo | 94.0 | 97.47 | +3.47 | +3.69 |
| cubo2 | ancho | 95.0 | 97.68 | +2.68 | +2.82 |
| cubo2 | alto | 95.0 | 95.85 | +0.85 | +0.89 |
| cubo3 | fondo | 94.0 | 97.66 | +3.66 | +3.89 |
| cubo3 | ancho | 95.0 | 98.04 | +3.04 | +3.20 |
| cubo3 | alto | 95.0 | 95.98 | +0.98 | +1.03 |
| piramide1 | lado_base | 103.0 | 98.82 | -4.18 | -4.06 |
| piramide1 | arista_lateral_hasta_punta | 100.0 | 96.65 | -3.35 | -3.35 |
| piramide1 | altura_cara | 85.0 | 83.05 | -1.95 | -2.29 |
| piramide3 | lado_base | 103.0 | 100.51 | -2.49 | -2.42 |
| piramide3 | arista_lateral_hasta_punta | 100.0 | 96.22 | -3.78 | -3.78 |
| piramide3 | altura_cara | 85.0 | 82.04 | -2.96 | -3.48 |
| piramide4 | lado_base | 103.0 | 99.39 | -3.61 | -3.50 |
| piramide4 | arista_lateral_hasta_punta | 100.0 | 95.16 | -4.84 | -4.84 |
| piramide4 | altura_cara | 85.0 | 81.14 | -3.86 | -4.55 |

Cuando falta una medida, la tabla indica **No disponible** y la excluye de las estadísticas. Para calcular la desviación estándar se necesitan al menos dos campañas válidas. El motivo de cada ausencia queda registrado en CSV y JSON.

## Variación entre campañas

| Objeto | Magnitud | Campañas válidas/esperadas | Media (mm) | Desviación estándar (mm) | Discrepancia absoluta media (mm) |
| --- | --- | ---: | ---: | ---: | ---: |
| cilindro | altura | 3/3 | 98.72 | 0.67 | 1.72 |
| cilindro | diametro | 3/3 | 44.99 | 0.54 | 1.99 |
| cubo | alto | 3/3 | 95.76 | 0.27 | 0.76 |
| cubo | ancho | 3/3 | 98.21 | 0.63 | 3.21 |
| cubo | fondo | 3/3 | 97.85 | 0.50 | 3.85 |
| piramide | altura_cara | 3/3 | 82.08 | 0.96 | 2.92 |
| piramide | arista_lateral_hasta_punta | 3/3 | 96.01 | 0.77 | 3.99 |
| piramide | lado_base | 3/3 | 99.57 | 0.86 | 3.43 |

El [CSV](comparacion_dimensional.csv) incluye todas las medidas de las mallas finales y previas al pulido. El [JSON](comparacion_dimensional.json) conserva métodos, diagnósticos y huellas. El modelo previo al pulido puede contener relleno anterior: esta comparación no separa observación de inferencia.

Reproducir: `python herramientas/comparar_dimensiones.py`. No actualiza el estado de aceptación ni activa calibraciones. Las tablas son resultados numéricos del repositorio; su discusión y conclusiones corresponden al documento de tesis.
