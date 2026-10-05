# Resultados y validación

[Volver al README](../README.md)

## Muestra publicada

[resultados/](../resultados/README.md) contiene nueve campañas seleccionadas: cilindros 1, 2 y 3; cubos 1, 2 y 3; pirámides 1, 3 y 4. El cubo 3 procede de `ubo3_20260919_131506`, nombre original con error de escritura. Incluye PLY final y previo al pulido, un par de imágenes por campaña, preview, métricas y procedencia. La calidad y las métricas corresponden a las ejecuciones identificadas en la procedencia de cada campaña. Los 75 pares originales por campaña no se duplican en el paquete público.

La selección pesa aproximadamente 165 MB. `MANIFEST.json` permite comprobar sus archivos y excluye su propia huella. Los JSON públicos omiten rutas absolutas personales; las métricas y advertencias no se cambian. La [comparación dimensional exploratoria](../resultados/comparacion_dimensional.md) aplica las [referencias aproximadas con regla](medidas_fisicas.md) a las nueve campañas, tanto a la malla final como a la previa al pulido. Explica los métodos y sus limitaciones; para la pirámide triangular compara estimaciones de lados de base, aristas y alturas de cara, conservando los puntos identificados y sin atribuir correspondencia individual con las caras físicas. Los 85 mm son altura de cara, no altura perpendicular.

Las exportaciones generadas por esta versión excluyen `resumen_exportacion_18.json` de su propio listado de archivos; su SHA-256 se guarda externamente en el resumen de la etapa 18. La [guía de operación](operacion_y_cierre.md) describe reparación histórica, respaldos y recuperación.

La [guía de medición dimensional](revision_medicion_dimensional.md) describe los descriptores utilizados y su sensibilidad. Distingue las discrepancias observadas de sus posibles causas, sin modificar las mallas ni las medidas publicadas.

## De la observación a la superficie

La profundidad inicial, la profundidad consensuada, la nube fusionada y la malla representan productos diferentes. El filtrado puede descartar observaciones y la reconstrucción de superficie puede interpolar zonas sin mediciones.

## Archivos de exportación

El paso 18 prepara `resultado_final/` tras las validaciones correspondientes.

| Archivo | Interpretación |
| --- | --- |
| `modelo_final.obj` | Modelo orientado para Blender, con coordenadas en metros y Z vertical |
| `modelo_final.mtl` | Material asociado al OBJ; mantener junto a él |
| `modelo_final_metric_mm.obj` | Geometría final en milímetros y coordenadas originales |
| `modelo_final_original_mm.ply` | Copia de la malla final métrica, conservando sus atributos de color |
| `modelo_pre_pulido_mm.ply` | Referencia anterior al pulido para comparación |
| `modelo_pre_pulido_metric_mm.obj` | Versión OBJ de esa referencia, en milímetros |
| `preview_validacion.png` | Vista de revisión de la validación |
| `resumen_exportacion_18.json` | Trazabilidad de fuentes y exportaciones |
| `README_IMPORTAR_EN_BLENDER.txt` | Instrucciones de importación generadas para el resultado |

La transformación de la exportación para Blender es `(X, -Z, Y) × 0.001` respecto a las coordenadas originales en milímetros. No aplique de nuevo la conversión de milímetros a metros al OBJ preparado para Blender. Para análisis métricos en el marco original, utilice los archivos identificados con `mm`.

## Estados de calidad

| Estado | Lectura |
| --- | --- |
| `accepted` | Cumple los controles de la etapa; no significa que la superficie esté completamente cerrada |
| `warning` | Hay advertencias que deben consultarse; puede incluir geometría estimada o una base abierta |
| `rejected` | No supera los controles; revisar las causas antes de usarlo como resultado válido |

El estado general de una etapa y la aceptación de un candidato de relleno son decisiones distintas. La etapa puede conservar una malla válida después de rechazar un candidato que la empeora. La exportación admite resultados aceptados o con advertencias conforme a sus controles; los rechazados se bloquean.

## Métricas

- **Distancia nube → malla:** separación de las observaciones respecto a la superficie. Un valor pequeño indica proximidad a los puntos disponibles.
- **Distancia malla → nube:** separación de la superficie respecto a las observaciones; ayuda a identificar regiones con poco respaldo.
- **P95:** percentil 95 de las distancias, expresadas en milímetros cuando el campo termina en `_mm`. No es el máximo.
- **Cobertura dentro de una tolerancia:** fracción del conjunto evaluado que queda a esa distancia. Una cobertura de puntos del 100 % no implica reconstruir el 100 % del objeto.
- **Bordes abiertos:** contornos sin cierre. Un contorno de base puede ser intencional; los laterales requieren revisión.
- **Intersecciones:** deben interpretarse con la clasificación de 16, que distingue contactos y cruces reales.

Ninguna métrica aislada garantiza dimensiones correctas. Una superficie puede aproximarse bien a una nube que ya contiene un sesgo de profundidad o registro.

## Relleno y tapa inferior

El paso 13 evalúa reconstrucciones de superficie y operaciones de completado. Cuando se acepta el candidato de paredes continuas, puede aparecer `selected_method = continuous_walls_open_base`. La base inferior se puede mantener abierta.

La aplicación del relleno depende de su fidelidad a los puntos observados y de los controles topológicos. Si el candidato no los cumple, se conserva la superficie anterior. El estado general de la etapa no indica por sí solo si se aplicó un relleno.

## Comparar dimensiones

1. Compare archivos en las mismas unidades y sistema de coordenadas.
2. Identifique si cada medición corresponde a la nube observada, a la malla o a una superficie completada.
3. Utilice el mismo criterio para altura, diámetro o aristas en ambas campañas.
4. Compare con medidas físicas del objeto y registre la diferencia.
5. Revise por separado el efecto del pulido y del relleno.

Las extensiones de una caja alineada con los ejes no son necesariamente las aristas de un cubo girado ni el diámetro de un cilindro inclinado. Una base abierta también impide interpretar directamente el volumen como el de un sólido cerrado.

## Evidencia conservada en modo reducido

Las ejecuciones de esta versión conservan, además de `resultado_final/`:

- Paso 12: `nube_regularizada_general.npz`, referencia de la validación final.
- Paso 13: `malla_observacional_antes_relleno.ply` cuando se realiza un intento de relleno, `malla_final_seleccionada.ply` y `procedencia_estimada.npz` cuando hay caras estimadas.
- Pasos 14 y 15: las mallas antes y después del pulido.

`procedencia_estimada.npz` identifica caras de la malla seleccionada del paso 13 y guarda su SHA-256. Sus índices no se deben aplicar directamente a mallas posteriores, cuya topología puede cambiar. La malla «observacional» es una reconstrucción anterior al completado, no una medición directa ni una verdad terreno.

La exportación prepulido corresponde al paso 14: puede contener relleno previo del paso 13. Para separar el efecto del relleno y el del pulido hay que usar ambas comparaciones.

La compactación registra los archivos retirados por etapa en `documentacion/resumen_almacenamiento.json` sin reescribir los informes científicos. La evidencia retirada en compactaciones históricas no forma parte de sus productos conservados; cambiar el modo de almacenamiento no restaura esos archivos.

## Evidencia de la implementación

La [guía de implementación y evidencia](implementacion_y_evidencia.md) reúne contratos de etapas, controles de consenso y superficie, indicadores de plataforma y registro, parámetros efectivos, capturas de interfaz y tiempos de una ejecución histórica identificada.
