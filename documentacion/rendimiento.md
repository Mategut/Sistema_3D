# Rendimiento y recursos

[Volver al README](../README.md)

## Distribución del procesamiento

El sistema combina inferencia en GPU con procesamiento paralelo en CPU. La estrategia depende de la etapa y del volumen de datos disponible.

| Recurso | Trabajo principal |
| --- | --- |
| GPU | Estimación de profundidad CREStereo mediante ONNX Runtime, cuando CUDA está disponible |
| CPU | Preparación de imágenes, máscaras, validación, consenso y operaciones geométricas |
| Memoria RAM | Imágenes, mapas de profundidad, nubes y estructuras de consulta |
| Disco | Capturas, referencias, productos intermedios y modelos finales |

La ejecución aprovecha tareas independientes por vistas o bloques. La concurrencia se ajusta a la CPU, la memoria disponible y la cantidad de trabajo. Las fases secuenciales y las transferencias de datos hacen que la utilización de CPU y GPU varíe durante la reconstrucción.

## Aceleración de profundidad

El proveedor `auto` selecciona un proveedor disponible y permite ejecución por CPU cuando no hay aceleración compatible. El proveedor `cuda` solicita explícitamente la GPU NVIDIA. La configuración del entorno se describe en [Instalación y uso](instalacion_y_uso.md).

Las operaciones geométricas de Open3D dependen de la implementación disponible; la aceleración de la inferencia no implica que todas las etapas se ejecuten en GPU.

## Configuración avanzada de CPU

La configuración automática es suficiente para el uso habitual. Las siguientes variables permiten ajustar las rutas que utilizan las utilidades de rendimiento:

| Variable | Función |
| --- | --- |
| `SISTEMA3D_CPU_THREADS` | Presupuesto de hilos; por defecto utiliza el número de procesadores lógicos |
| `SISTEMA3D_WORKERS` | Número solicitado de trabajadores |
| `SISTEMA3D_BLAS_THREADS` | Hilos internos de las bibliotecas numéricas; predeterminado: 1 por proceso |
| `SISTEMA3D_WORKER_RESERVE_MB` | Reserva adicional de memoria considerada por trabajador |

Cada etapa conserva sus límites de concurrencia y memoria. El paso 04 dispone además de su propia configuración de trabajadores. El límite de hilos internos evita que cada proceso cree un grupo completo de hilos y sobrecargue el equipo.

## Almacenamiento

| Modo | Uso |
| --- | --- |
| `reducido` | Ejecución habitual: conserva capturas, referencias, informes, resultados finales y evidencia geométrica de validación y relleno |
| `completo` | Análisis detallado: conserva también los intermedios por vista y etapa |

El modo se selecciona con `--storage-mode`. Conservar todos los intermedios aumenta el espacio ocupado. Si ya fueron retirados, es necesario recalcularlos para volver a disponer de ellos.

## Tiempos de ejecución

Los registros incluyen tiempos por etapa. Los archivos de `registros/rendimiento/` añaden muestreos de CPU, memoria y GPU cuando están disponibles; parte de las mediciones corresponde al equipo completo.

El tiempo depende de la cantidad de vistas y puntos, la complejidad geométrica y las operaciones de reconstrucción. Para comparar rendimiento, utilice la misma campaña, referencias y configuración, distinguiendo una ejecución completa de una reanudación.
