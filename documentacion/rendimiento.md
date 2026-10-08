# Rendimiento y recursos

[Volver al README](../README.md)

## Distribución del procesamiento

La GPU puede ejecutar la estimación de profundidad; las demás etapas realizan trabajo en CPU. La distribución de tareas varía según la operación y el volumen de datos.

| Recurso | Trabajo principal |
| --- | --- |
| GPU | Estimación de profundidad CREStereo mediante ONNX Runtime, cuando CUDA está disponible |
| CPU | Preparación de imágenes, máscaras, validación, consenso y operaciones geométricas |
| Memoria RAM | Imágenes, mapas de profundidad, nubes y estructuras de consulta |
| Disco | Capturas, referencias, productos intermedios y modelos finales |

Las tareas independientes se distribuyen por vistas o bloques, con una concurrencia ajustada a la CPU, la memoria disponible y la cantidad de trabajo. Otras fases se ejecutan en secuencia y requieren transferencias de datos. Por eso, el uso de CPU y GPU varía durante la reconstrucción.

## Aceleración de profundidad

La opción `auto` selecciona un proveedor disponible y permite ejecutar en CPU cuando no hay aceleración compatible. El proveedor `cuda` solicita explícitamente la GPU NVIDIA. La configuración del entorno se describe en [Instalación y uso](instalacion_y_uso.md).

Las operaciones geométricas de Open3D dependen de la implementación disponible; la aceleración de la inferencia no implica que todas las etapas se ejecuten en GPU.

## Configuración avanzada de CPU

La configuración automática es suficiente para el uso habitual. Las siguientes variables permiten ajustar las rutas que utilizan las utilidades de rendimiento:

| Variable | Función |
| --- | --- |
| `SISTEMA3D_CPU_THREADS` | Presupuesto de hilos; por defecto utiliza el número de procesadores lógicos |
| `SISTEMA3D_WORKERS` | Número solicitado de trabajadores |
| `SISTEMA3D_BLAS_THREADS` | Hilos internos de las bibliotecas numéricas; predeterminado: 1 por proceso |
| `SISTEMA3D_WORKER_RESERVE_MB` | Reserva adicional de memoria considerada por trabajador |

Cada etapa conserva sus límites de concurrencia y memoria. El paso 04 dispone además de su propia configuración de trabajadores. Limitar los hilos internos evita que los procesos compitan por todos los recursos del equipo.

## Almacenamiento

| Modo | Uso |
| --- | --- |
| `reducido` | Ejecución habitual: conserva capturas, referencias, informes, resultados finales y evidencia geométrica de validación y relleno |
| `completo` | Análisis detallado: conserva también los intermedios por vista y etapa |

El modo se selecciona con `--storage-mode`. Conservar todos los intermedios aumenta el espacio ocupado. Si ya fueron retirados, es necesario recalcularlos para volver a disponer de ellos.

## Tiempos de ejecución

Los registros incluyen tiempos por etapa. Los archivos de `registros/rendimiento/` añaden muestreos de CPU, memoria y GPU cuando están disponibles; parte de las mediciones corresponde al equipo completo.

El tiempo de ejecución depende de la cantidad de vistas y puntos, la complejidad geométrica y las operaciones de reconstrucción. Para comparar rendimiento, mantenga la campaña, las referencias y la configuración, e identifique si se ejecutó todo el procesamiento o se reutilizaron etapas al reanudar.

## Evidencia de la implementación

La [guía de implementación y evidencia](implementacion_y_evidencia.md) presenta los tiempos por etapa de una ejecución histórica y distingue sus intervalos internos y externos, junto con las fuentes de telemetría.
