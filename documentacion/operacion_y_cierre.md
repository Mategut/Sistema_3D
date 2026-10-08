# Operación, parámetros y entrega del proyecto

## Recorrido de uso

1. Instale el entorno según [instalación y uso](instalacion_y_uso.md). Ejecute la verificación de dependencias y revise el resultado completo.
2. Compruebe el montaje de cámaras y plataforma, los índices de cámara y el puerto serie. El firmware corresponde a 2055 pasos distribuidos entre 25 posiciones.
3. Prepare o importe una calibración estéreo correspondiente a la geometría física actual. Verifique estéreo y capture fondo vacío después de estabilizar enfoque, exposición e iluminación.
4. Capture y procese la campaña de calibración de plataforma. Al aprobar el paso 09, la aplicación instala el resultado con respaldo: siga la [guía de calibración](validacion_independiente_plataforma.md).
5. Cree una campaña por objeto, capture las tres sesiones y procese. Mantenga las referencias congeladas; modificar recursos globales no cambia las referencias de un trabajo existente.
6. Consulte calidad, advertencias, cierre y soporte observacional antes de usar los modelos. Abra los archivos de `resultado_final/` y compruebe las unidades de importación.

## Parámetros que condicionan la interpretación

| Grupo | Dónde se configura | Efecto y criterio |
| --- | --- | --- |
| Esquinas internas y lado del cuadro | Formulario estéreo | Deben coincidir con el tablero físico; un lado incorrecto afecta la escala métrica. |
| Separación aproximada y tolerancia | Formulario estéreo | Control de coherencia del montaje; no reemplaza la estimación ni una medición del patrón. |
| Cámaras, resolución y fondo | Captura y referencias | El montaje utiliza 1920 × 1080. Si cambia la geometría, recalibre; si cambia el fondo o iluminación, actualice su captura. |
| Motor de inferencia | Herramientas, `--provider` | `auto` permite recurrir a otro proveedor disponible; los motores explícitos exigen disponibilidad. El proveedor visible no garantiza bibliotecas CUDA operativas. |
| Conservación | Herramientas, `--storage-mode` | `completo` conserva diagnóstico intermedio; `reducido` conserva los productos científicos definidos. No recupera archivos previamente eliminados. |
| Máscaras, consistencia LR y soporte | Pasos 02–06 | Determinan qué observaciones sustentan la geometría. Para una evaluación independiente, conserve criterios comunes entre objetos; ajustarlos a cada resultado altera la comparación. |
| Registro y fusión | Pasos 08/10–12 | Afectan alineamiento y densidad. Diferencie error de registro de error de superficie. |
| Completado y pulido | Pasos 13–15 | Pueden inferir superficie. El modelo final y el previo al pulido no equivalen automáticamente a observaciones directas. |
| Cobertura y distancias | Paso 17 | Cobertura: radio declarado 3 mm, advertencia 0.95, rechazo 0.85. P90 nube→malla: 2.5/4.5 mm. P90 malla→nube: 2.0/4.0 mm, con adaptación activada y techo de rechazo 6 mm. Consulte los umbrales efectivos del informe. |
| Exportación | Paso 18 | Hereda calidad; una exportación con warning debe conservar las advertencias. OBJ Blender usa metros; OBJ científico y PLY usan milímetros. |

La [referencia completa de argumentos](referencia_parametros.md) se genera del código. Incluye valores declarados, opciones y ayuda por etapa. Para interpretar una ejecución, consulte primero los argumentos del coordinador y sus informes: un valor predeterminado aislado puede haber sido sustituido.

## Fallos y recuperación

| Síntoma | Acción |
| --- | --- |
| Cámara ocupada o índice incorrecto | Cierre la aplicación que la utiliza y revise índices. La captura del tablero libera las cámaras de la vista previa; reconecte al terminar. |
| Puerto serie sin respuesta | Compruebe puerto, baudios 115200, cable y firmware. No cambie la posición inicial física durante una captura. |
| Error CUDA o proveedor no disponible | Revise el diagnóstico. Use `auto` o `cpu` desde Herramientas si corresponde. Consulte el registro para identificar el proveedor utilizado. |
| Huellas de referencias distintas | Conserve el diagnóstico. Restaure los bytes originales o cree otra campaña con las referencias correctas. Las huellas deben corresponder a los archivos utilizados. |
| Captura interrumpida | Reinicie la captura con el objeto en su posición inicial. El reinicio lógico del Arduino no mueve físicamente el montaje a un origen. |
| Procesamiento interrumpido | Reabra el trabajo y reanude. Solo se reutilizan checkpoints vigentes; si cambian código, entradas o productos, puede ser necesario recalcular. |
| Candidata o informe rechazados | Lea el registro y corrija el experimento o los archivos. Guardar un informe no implica verificarlo ni activar una calibración. |
| Exportación rechazada | Revise las causas en los pasos 16 y 17 y corrija el problema antes de exportar. El estado de calidad debe proceder de la validación. |
| Falta de evidencia LR en un trabajo reducido | Vuelva a procesar con conservación completa. La compactación anterior no es reversible sin los datos originales. |

`Parar` solicita cancelar el proceso y detener el motor. El firmware no interrumpe un movimiento bloqueante ya iniciado. Espere a la confirmación de parada antes de manipular el montaje.

## Integridad de exportación

El informe `resumen_exportacion_18.json` registra las huellas de los archivos exportados y excluye su propio nombre. Para evitar la autorreferencia, su huella se guarda por separado como `export_report_sha256` en el resumen de la etapa 18.

La reparación histórica se ejecuta con `python herramientas/reparar_manifiestos_exportacion.py`. Primero comprueba las huellas de los archivos exportados y detiene la reparación si encuentra diferencias. Conserva los informes anteriores en `registros/reparacion_manifiestos_*/` y corrige únicamente el manifiesto y su resumen asociado. No modifica mallas ni métricas. La reparación no renueva los checkpoints: el coordinador vuelve a comprobarlos al reanudar.

## Contenido de la entrega

- Código, firmware, configuración y documentación del proyecto.
- `modelos/`: ONNX, identidad técnica, referencias, licencia upstream y aclaración de procedencia.
- `sistema/`: referencias del montaje de desarrollo, conservadas byte a byte por Git; no son una calibración universal para otros montajes.
- `resultados/`: nueve campañas seleccionadas, con originales PLY, imágenes, advertencias, procedencia y manifiesto independiente.
- Se excluyen `trabajos/`, `registros/`, `respaldos/`, `backups/`, cachés y archivos temporales. No elimine las campañas originales: son necesarias para una reproducción completa.

La muestra pública permite inspeccionar los resultados. Repetir el procesamiento completo requiere también las capturas y sus referencias originales. El script de selección trabaja con campañas identificadas y exige una carpeta de salida nueva.

`resultados/MANIFEST.json` registra las huellas de los archivos, salvo la del propio manifiesto. Si modifica un documento publicado o sus finales de línea, regenere el manifiesto para que siga describiendo el contenido del paquete.

## Alcance de la entrega

La entrega reúne el software, el firmware, las referencias del montaje y la evidencia de nueve campañas de cubo, cilindro y pirámide. La documentación describe instalación, interfaz, arquitectura, parámetros, almacenamiento, resultados, integridad, reproducción y referencias del modelo. Los estados de aceptación describen los controles internos registrados en cada ejecución, no una certificación metrológica.

Se incluye una [comparación dimensional exploratoria](../resultados/comparacion_dimensional.md) con las referencias físicas aproximadas tomadas con regla. La autoría, el título y la institución están documentados en el [README principal](../README.md#identificación-del-proyecto); el montaje y el equipo de cómputo de referencia, en [instalación y uso](instalacion_y_uso.md#montaje-de-referencia). La conversión ONNX utilizada procede de PINTO Model Zoo, sin que se haya comprobado la igualdad binaria con una descarga upstream.

El alcance experimental se limita a los tres tipos de objeto de la muestra y al montaje utilizado. Las referencias físicas son aproximadas, carecen de incertidumbre cuantificada y no establecen una correspondencia individual entre todos los segmentos medidos y los estimados. El catálogo y los archivos de procedencia identifican las ejecuciones históricas que sustentan los resultados publicados.

El código propio y su documentación de software se publican bajo [licencia MIT](../LICENSE), con el alcance y las excepciones de terceros descritos en el [README](../README.md#licencia).

## Controles de importación y exportación

El iniciador usa `verificar_dependencias.py --startup`: una calibración estéreo ausente o inválida se comunica como advertencia y permite abrir la interfaz para repararla. La verificación normal continúa fallando ante ese problema. La aplicación comprueba el contenido antes de permitir la reconstrucción o instalar una calibración: informe aceptado sin causas de rechazo, mapas finitos de tamaño correcto y coherencia de matrices, baseline y RMS.

El paso 18 exige calidad `accepted` o `warning`, una lista vacía de causas de rechazo y la huella SHA-256 de la malla validada en el paso 17. Los informes históricos sin esa huella requieren ejecutar nuevamente el paso 17 antes de una nueva exportación; no se les atribuye retroactivamente una validación. Los resultados históricos publicados se conservan como evidencia de su ejecución original.

La política del paso 17 declara que superar el umbral efectivo de rechazo P90 malla→nube siempre rechaza el resultado, incluso con coherencia independiente favorable. Los informes antiguos que declaraban una excepción contenían una descripción contradictoria con el control ejecutado; sus métricas y estados no se han reescrito.

Al reemplazar `resultado_final`, el paso 18 conserva la entrega previa en `registros_exportacion/resultado_anterior_<marca_temporal>` dentro de la campaña. Si falla el reemplazo del directorio temporal, intenta restaurar la entrega anterior. Los respaldos permanecen disponibles después de una publicación correcta.

La validación estructural estéreo también se ejecuta antes de procesar, desde interfaz y coordinador, sobre las referencias efectivamente utilizadas por la campaña. Una referencia histórica inválida se bloquea; no se sustituye automáticamente.

La exportación registra el reemplazo en `resultado_final_publicacion.json`. Antes de otra exportación, restaura el respaldo si falta la entrega, o verifica el manifiesto de la nueva entrega si ya fue publicada. Si encuentra una situación ambigua conserva los archivos y solicita revisión. Una interrupción antes de terminar la exportación puede exigir repetir el paso 18 para completar su resumen.

## Evidencia de la implementación

La [guía de implementación y evidencia](implementacion_y_evidencia.md) permite relacionar el recorrido operativo con las capturas de la interfaz, los parámetros y los productos de procesamiento conservados.
