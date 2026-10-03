# Sistema de reconstrucción 3D por visión estereoscópica

Software de un proyecto de grado para reconstruir objetos mediante dos cámaras web y una plataforma giratoria controlada por Arduino. Integra adquisición, estimación de profundidad, registro multivista, reconstrucción de superficie y exportación en escala métrica.

Cada adquisición se organiza como una **campaña**, con sus capturas, referencias del montaje, resultados e informes de calidad. La configuración habitual utiliza tres sesiones de 25 posiciones: 75 pares estéreo.

## Explorar los modelos en 3D

[**Abrir la galería interactiva de resultados**](https://mategut.github.io/Sistema_3D/)

Permite girar y ampliar las nueve reconstrucciones, alternar entre la malla final y la anterior al pulido, consultar métricas y descargar los PLY originales. La galería está publicada en GitHub Pages y puede consultarse sin instalar la aplicación. Los controles y la interpretación de los resultados se explican en la [guía del visor](visor/LEEME.md).

## Identificación del proyecto

- **Autor:** Mateo Gutiérrez Mejía.
- **Trabajo de grado:** *Aplicación de software en Python para reconstrucción tridimensional de bajo costo basada en visión estereoscópica con objeto rotatorio y dos cámaras web*.
- **Institución:** Universidad Santo Tomás, Facultad de Ingeniería Electrónica, División de Ingenierías, Bogotá D.C.; Grupo de Estudio y Desarrollo en Robótica (GED).
- **Director:** Billy Wladimir Toro Tovar, Ph.D. **Codirector:** Armando Mateus Rojas, M.Sc.
- **Repositorio:** [Mategut/Sistema_3D](https://github.com/Mategut/Sistema_3D).

Este repositorio constituye la entrega de software y evidencia del proyecto: incluye el código, el firmware, las referencias del montaje, el modelo de profundidad, los manuales y nueve campañas documentadas de cubo, cilindro y pirámide. Los modelos publicados conservan las métricas y los estados de calidad registrados en sus ejecuciones. El documento académico de tesis desarrolla la fundamentación y la discusión de los resultados y no forma parte de este paquete.

La [guía de instalación](documentacion/instalacion_y_uso.md#montaje-de-referencia) incluye los esquemas del banco, el cableado Arduino–ULN2003–28BYJ-48 y el equipo de cómputo. El alcance experimental y las limitaciones de la evidencia se describen en [Resultados y validación](documentacion/resultados_y_validacion.md).

## Documentación

| Guía | Contenido |
| --- | --- |
| [Manual de instalación y uso](documentacion/instalacion_y_uso.md) | Equipo de referencia, montaje, calibración, recorrido por los botones, resultados y reanudación |
| [Arquitectura y código](documentacion/arquitectura_y_codigo.md) | Responsabilidad de cada script y organización de módulos |
| [Resultados y validación](documentacion/resultados_y_validacion.md) | Exportaciones, unidades, métricas y relleno |
| [Rendimiento](documentacion/rendimiento.md) | Procesamiento en CPU y GPU, memoria y almacenamiento |
| [Operación y cierre](documentacion/operacion_y_cierre.md) | Parámetros decisivos, recuperación, integridad y alcance de la entrega |
| [Referencia de parámetros](documentacion/referencia_parametros.md) | Argumentos y valores declarados por etapa |
| [Resultados seleccionados](resultados/README.md) | Nueve campañas con modelos, imágenes, métricas y advertencias |
| [Modelo y atribuciones](modelos/README.md) | Identidad del ONNX, procedencia conocida, licencia upstream y citación |

## Versión del software

La versión de esta entrega es **3.2.1**. Su fuente única es [version_sistema.py](procesamiento/version_sistema.py); la interfaz, el coordinador y los informes de calibración y exportación generados por esta versión utilizan esa constante. Las versiones de esquema y de métodos tienen significado independiente. Los informes históricos conservan la versión con la que fueron generados.

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

## Licencia

El código propio de Sistema 3D y su documentación de software se distribuyen bajo la [licencia MIT](LICENSE), con copyright © 2026 Mateo Gutiérrez Mejía. Permite utilizar, modificar y redistribuir el software, incluso comercialmente, conservando el aviso de autoría y la licencia. Se proporciona sin garantía.

Los componentes de terceros conservan sus licencias y atribuciones: consulte los [avisos de CREStereo y del modelo ONNX](modelos/NOTICE.md) y la [licencia de Three.js](visor/vendor/LICENSE). La licencia MIT del proyecto no sustituye esas condiciones ni atribuye MIT al modelo ONNX.

Esta licencia se refiere al software y su documentación asociada; no concede una licencia sobre el documento académico de tesis, logotipos institucionales, capturas, conjuntos de datos o modelos 3D de resultados. Su disponibilidad pública no implica que se les aplique MIT.
