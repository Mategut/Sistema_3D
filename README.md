# Sistema de reconstrucción 3D por visión estereoscópica

Software de un proyecto de grado para reconstruir objetos mediante dos cámaras web y una plataforma giratoria controlada por Arduino. Integra adquisición, estimación de profundidad, registro multivista, reconstrucción de superficie y exportación en escala métrica.

Cada adquisición se organiza como una **campaña**, con sus capturas, referencias del montaje, resultados e informes de calidad. La configuración habitual utiliza tres sesiones de 25 posiciones: 75 pares estéreo.

## Documentación

| Guía | Contenido |
| --- | --- |
| [Instalación y uso](documentacion/instalacion_y_uso.md) | Entorno, montaje, captura, comandos y reanudación |
| [Arquitectura y código](documentacion/arquitectura_y_codigo.md) | Responsabilidad de cada script y organización de módulos |
| [Resultados y validación](documentacion/resultados_y_validacion.md) | Exportaciones, unidades, métricas y relleno |
| [Rendimiento](documentacion/rendimiento.md) | Procesamiento en CPU y GPU, memoria y almacenamiento |
| [Operación y cierre](documentacion/operacion_y_cierre.md) | Parámetros decisivos, recuperación, integridad y alcance de la entrega |
| [Referencia de parámetros](documentacion/referencia_parametros.md) | Argumentos y valores declarados por etapa |
| [Resultados seleccionados](resultados/README.md) | Nueve campañas con modelos, imágenes, métricas y advertencias |
| [Modelo y atribuciones](modelos/README.md) | Identidad del ONNX, procedencia conocida, licencia upstream y citación |

## Requisitos

- Windows y Python 3.11 recomendado.
- Dos cámaras en un montaje estéreo fijo.
- Plataforma giratoria con Arduino y el [firmware incluido](firmware/control_plataforma_2055_pasos/control_plataforma_2055_pasos.ino).
- Modelo `crestereo_init_iter10_480x640.onnx` en `modelos/`.
- Calibraciones y fondo vacío correspondientes al montaje.
- Para acelerar la inferencia: GPU NVIDIA y entorno CUDA/cuDNN compatible con ONNX Runtime GPU. También existe ejecución por CPU.

Las dependencias están en [requirements.txt](requirements.txt). Las calibraciones del repositorio corresponden al montaje de desarrollo; deben revisarse antes de utilizarlas en otro equipo.

## Inicio rápido

Desde la raíz del proyecto, con el entorno Python activado:

```powershell
python -m pip install -r requirements.txt
python herramientas/verificar_dependencias.py --no-install
```

Después:

1. Preparar las cámaras y cargar el firmware en Arduino.
2. Comprobar las calibraciones y referencias del montaje.
3. Ejecutar `INICIAR_SISTEMA_3D.bat`.
4. Configurar cámaras y puerto serie, y actualizar el fondo vacío sin el objeto.
5. Crear la campaña, capturar las sesiones y ejecutar la reconstrucción.
6. Revisar los informes de calidad y `resultado_final/`.

`VERIFICAR_SISTEMA.bat` intenta instalar las dependencias faltantes. La opción `--no-install` permite comprobarlas sin instalar. El lanzador busca preferentemente el entorno Conda `tesis`; para usar el intérprete activo puede ejecutar `python Sistema_3D.py`.

La interfaz organiza las acciones en **Captura**, **Calibración** y **Herramientas**. Permite crear y verificar la calibración estéreo e instalar automáticamente la calibración de plataforma al aprobar el paso 09, sin informes externos ni PowerShell. Incluye diagnósticos, registros y selección de motor y almacenamiento. Consulte el [flujo de calibración en la interfaz](documentacion/instalacion_y_uso.md#calibraciones-desde-la-interfaz).

## Organización

```text
Sistema_3D.py                 Interfaz, adquisición y control general
INICIAR_SISTEMA_3D.bat        Lanzador
VERIFICAR_SISTEMA.bat         Verificación del entorno
firmware/                    Control de la plataforma
herramientas/                Calibración y auditoría
modelos/                     Modelo ONNX
procesamiento/               Coordinador, etapas 01–18 y utilidades
sistema/                     Referencias activas del montaje
documentacion/               Guías de uso y referencia técnica
resultados/                  Nueve campañas seleccionadas y comparación dimensional
trabajos/<campaña>/
  capturas/                  Pares estéreo originales
  documentacion/             Configuración y referencias congeladas
  reconstruccion/            Intermedios, informes y checkpoints
  resultado_final/           Exportación validada
registros/                   Logs y rendimiento
```

`trabajos/` y `registros/` están excluidos de Git. Las referencias globales de `sistema/` sirven para nuevas campañas; las existentes conservan sus copias en `documentacion/referencias/`.

Consulte la [guía de referencias del montaje](sistema/README.md) para conocer el contenido de `sistema/` y cuándo actualizarlo. La [guía de arquitectura](documentacion/arquitectura_y_codigo.md#herramientas-por-finalidad) organiza las herramientas según su finalidad y frecuencia de uso.

## Flujo de procesamiento

Reconstrucción normal:

```text
01 → 02 → 03 → 04 → 05 → 06 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18
```

Calibración de plataforma:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

El [coordinador 00](procesamiento/00_ejecutar_pipeline.py) ejecuta la ruta y permite reanudar mediante checkpoints. Los pasos 07–09 pertenecen a la calibración y no se repiten en cada reconstrucción normal.

## Resultados

La exportación genera un OBJ en metros para Blender y versiones OBJ y PLY en milímetros. El PLY conserva los atributos de color de la malla. Cada campaña incluye informes de calidad y una referencia anterior al pulido para comparar la geometría.

La superficie puede incorporar regiones estimadas y conservar la base inferior abierta. La [guía de resultados](documentacion/resultados_y_validacion.md) explica las unidades, las métricas y los estados de validación.

## Almacenamiento

El modo predeterminado `reducido` conserva capturas, referencias, resultados finales, informes y evidencia geométrica de validación y relleno; retira otros intermedios pesados al finalizar. El modo `completo` conserva también los productos de cada etapa.

Al terminar correctamente el paso 09 desde la aplicación, se instala la calibración de plataforma y se respalda la anterior. Consulte el [flujo de calibración de plataforma](documentacion/validacion_independiente_plataforma.md).
