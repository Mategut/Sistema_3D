# Sistema de reconstrucción 3D por visión estereoscópica

Este repositorio contiene el software desarrollado para mi proyecto de grado de reconstrucción tridimensional de bajo costo mediante dos cámaras web y una plataforma giratoria controlada.

El sistema captura pares estereoscópicos en múltiples posiciones angulares, estima profundidad, genera nubes de puntos por vista, realiza registro y fusión multivista, reconstruye la superficie y exporta el modelo final en escala métrica.

## Requisitos

- Windows.
- Python 3.11 recomendado.
- Arduino con el firmware incluido en `firmware/`.
- Dos cámaras web configuradas para captura estéreo.
- GPU NVIDIA compatible con CUDA para ejecutar CREStereo mediante ONNX Runtime GPU.

Las dependencias están definidas en `requirements.txt`.

## Inicio

1. Ejecutar `VERIFICAR_SISTEMA.bat`.
   - El verificador comprueba las dependencias del entorno.
   - Si falta un paquete requerido, intenta instalarlo automáticamente con el mismo Python utilizado por el sistema.
   - La disponibilidad de CUDA/cuDNN no puede resolverse únicamente con `pip`; si ONNX Runtime no detecta `CUDAExecutionProvider`, el verificador lo reportará.
2. Ejecutar `INICIAR_SISTEMA_3D.bat`.
3. Actualizar el fondo vacío cuando cambie físicamente el montaje.
4. Crear un trabajo desde la interfaz y realizar la captura o reconstrucción correspondiente.

Las carpetas `trabajos/` y `registros/` se crean automáticamente y no se versionan en Git.

## Estructura

```text
Sistema_3D.py                 Interfaz, captura y control general
firmware/                     Control de la plataforma giratoria
herramientas/                 Verificación y calibración estéreo
modelos/                      Modelo ONNX de CREStereo
procesamiento/                Pipeline de reconstrucción 01-18
sistema/
  calibracion_estereo/        Calibración estéreo activa para nuevas campañas
  calibracion_plataforma/     Calibración activa para nuevas campañas
  fondo_vacio/                Fondo vigente para nuevas campañas
trabajos/
  <campaña>/
    capturas/                 Tres sesiones de captura
    documentacion/
        referencias/          Snapshot inmutable de los recursos de esa campaña
        configuracion_captura.json
        referencias_campana.json
    reconstruccion/           Productos intermedios y checkpoints
    resultado_final/          Modelo final
```

## Referencias congeladas por campaña

Cada trabajo conserva dentro de su propia carpeta `documentacion/` una copia de las referencias con las que fue creado. Esa carpeta pertenece al trabajo, no al repositorio, y permite reprocesar la campaña sin depender de cambios posteriores del montaje o de los recursos globales.

## Pipeline

Reconstrucción normal:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18
```

Calibración de plataforma:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09
```

`procesamiento/00_ejecutar_pipeline.py` coordina las etapas y permite reanudar ejecuciones mediante checkpoints. Cuando el trabajo contiene referencias congeladas, el coordinador las resuelve automáticamente aunque los argumentos de línea de comandos apunten a los recursos globales.

En una campaña de calibración de plataforma, el resultado del paso 09 se guarda primero dentro del propio trabajo en `resultado_calibracion_plataforma/`. Solo después de terminar correctamente se promociona como calibración activa del sistema, conservando la anterior en `registros/`.

El paso 04 valida la profundidad observada dentro de la silueta del paso 03: descarta píxeles sin disparidad válida, fuera del dominio rectificado o fuera del rango físico de profundidad. La consistencia entre sesiones continúa en el paso 05. Como control visual, el paso genera una hoja de contacto con todas las vistas procesadas.

## Criterio de diseño

El pipeline está planteado para trabajar con geometrías distintas sin imponer una forma conocida al objeto. Las decisiones de filtrado, registro, fusión y reconstrucción se basan en evidencia estéreo, consistencia multivista, incertidumbre y continuidad geométrica.

Las calibraciones incluidas corresponden al montaje utilizado durante el desarrollo. Si cambia la posición relativa entre las cámaras debe repetirse la calibración estéreo. Si el conjunto de cámaras cambia respecto a la plataforma, también debe actualizarse el fondo vacío y revisarse/repetirse la calibración de plataforma antes de crear nuevas campañas.

## Almacenamiento reducido

La aplicación conserva por defecto solo los productos científicos relevantes al terminar cada reconstrucción: modelos finales, resúmenes, tablas de calidad, hojas de contacto y previews globales. Los artefactos pesados por vista/pose se usan durante el cálculo y se retiran al finalizar. Las capturas originales, las referencias congeladas del trabajo y el resultado final no se eliminan.
