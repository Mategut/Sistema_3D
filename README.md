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
3. Crear un trabajo desde la interfaz y realizar la captura o reconstrucción correspondiente.

Las carpetas `trabajos/` y `registros/` se crean automáticamente y no se versionan en Git.

## Estructura

```text
Sistema_3D.py                 Interfaz, captura y control general
firmware/                     Control de la plataforma giratoria
herramientas/                 Verificación y calibración estéreo
modelos/                      Modelo ONNX de CREStereo
procesamiento/                Pipeline de reconstrucción 01-18
sistema/
  calibracion_estereo/        Calibración estéreo activa
  calibracion_plataforma/     Calibración activa del eje y plataforma
  fondo_vacio/                Fondos de referencia para segmentación
```

## Pipeline

Reconstrucción normal:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18
```

Calibración de plataforma:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09
```

`procesamiento/00_ejecutar_pipeline.py` coordina las etapas y permite reanudar ejecuciones mediante checkpoints.

## Criterio de diseño

El pipeline está planteado para trabajar con geometrías distintas sin imponer una forma conocida al objeto. Las decisiones de filtrado, registro, fusión y reconstrucción se basan en evidencia estéreo, consistencia multivista, incertidumbre y continuidad geométrica.

Las calibraciones incluidas corresponden al montaje utilizado durante el desarrollo. Si cambia la posición de las cámaras, la distancia entre ellas o la geometría de la plataforma, deben recalibrarse antes de realizar nuevas reconstrucciones.
