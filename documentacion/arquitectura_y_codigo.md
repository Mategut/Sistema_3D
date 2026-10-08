# Arquitectura y referencia del código

[Volver al README](../README.md)

## Arquitectura

La interfaz gestiona la captura y organiza los archivos de cada campaña. El coordinador toma esas entradas, ejecuta las etapas y comprueba sus productos; también guarda los checkpoints necesarios para reanudar el procesamiento. Las sesiones se procesan primero por separado. El consenso y las operaciones geométricas integran después sus resultados en `reconstruccion/multisesion/`.

```text
Capturas + referencias de campaña
               ↓
01 mapa angular → 02 profundidad → 03 máscara → 04 validación
               ↓
05 consenso entre sesiones → 06 nubes por pose
               ├─ Calibración: 07 → 08 → 09
               └─ Reconstrucción: 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18
```

La reconstrucción utiliza la calibración de plataforma disponible. Esa referencia se obtiene mediante una ruta de estimación y validación cuyos controles dependen de los supuestos geométricos del objeto de referencia.

## Aplicación

| Archivo | Responsabilidad |
| --- | --- |
| [Sistema_3D.py](../Sistema_3D.py) | Interfaz, cámaras, comunicación serie, campañas y lanzamiento del procesamiento |
| [interfaz_herramientas.py](../interfaz_herramientas.py) | Pestañas de calibración y diagnóstico, procesos en segundo plano e instalación de resultados guardados con respaldo |
| [INICIAR_SISTEMA_3D.bat](../INICIAR_SISTEMA_3D.bat) | Selección del entorno y arranque de la aplicación |
| [VERIFICAR_SISTEMA.bat](../VERIFICAR_SISTEMA.bat) | Verificación del entorno y calibración disponible |
| [Firmware](../firmware/control_plataforma_2055_pasos/control_plataforma_2055_pasos.ino) | Movimientos, contadores y protocolo de la plataforma |

## Herramientas por finalidad

Las utilidades de `herramientas/` permiten preparar el montaje, consultar diagnósticos y gestionar la publicación. Se agrupan según su finalidad; para capturar y reconstruir un objeto, siga el recorrido de la aplicación.

### Calibración y verificación del montaje

Se utilizan al preparar o modificar el montaje, y durante la evaluación de una candidata de plataforma. No es necesario ejecutar toda esta lista para cada objeto.

| Herramienta | Responsabilidad |
| --- | --- |
| [calibrar_estereo_checkerboard.py](../herramientas/calibrar_estereo_checkerboard.py) | Captura del patrón y calibración estéreo |
| [verificar_calibracion_estereo.py](../herramientas/verificar_calibracion_estereo.py) | Comprobación de los recursos de calibración |
| [promover_calibracion_plataforma.py](../herramientas/promover_calibracion_plataforma.py) | Instalación de una calibración aprobada por el paso 09; revisión independiente opcional por consola |
| [verificar_informe_calibracion.py](../herramientas/verificar_informe_calibracion.py) | Verificación independiente del informe sin activar ni sustituir referencias |

### Diagnóstico del entorno y de la evidencia

Útiles en la instalación y cuando se necesita investigar un problema de ejecución o de los datos.

| Herramienta | Responsabilidad |
| --- | --- |
| [registrar_entorno.py](../herramientas/registrar_entorno.py) | Registro del entorno Python, sistema, controlador NVIDIA e identidad del ONNX sin instalar paquetes ni ejecutar inferencia |
| [verificar_dependencias.py](../herramientas/verificar_dependencias.py) | Dependencias, proveedores y referencias estéreo; puede instalar paquetes faltantes. `--no-install` limita la acción a la comprobación. |
| [auditar_evidencia_lr.py](../herramientas/auditar_evidencia_lr.py) | Auditoría de evidencia y consistencia izquierda–derecha de una campaña |

### Preparación de resultados y documentación

Estas herramientas preparan los resultados públicos y sus documentos. Se ejecutan por separado del proceso de captura y reconstrucción.

| Herramienta | Responsabilidad |
| --- | --- |
| [preparar_resultados_publicos.py](../herramientas/preparar_resultados_publicos.py) | Selección pública reproducible, procedencia y anonimización de rutas |
| [comparar_dimensiones.py](../herramientas/comparar_dimensiones.py) | Comparación exploratoria de mallas con referencias físicas aproximadas; genera tablas, diagnósticos y actualiza el manifiesto público |
| [generar_referencia_parametros.py](../herramientas/generar_referencia_parametros.py) | Extracción de argumentos mediante AST sin ejecutar las etapas |

### Reparación histórica

Operación excepcional para exportaciones antiguas afectadas por la autorreferencia de sus manifiestos. No forma parte del flujo habitual de una campaña nueva.

| Herramienta | Responsabilidad |
| --- | --- |
| [reparar_manifiestos_exportacion.py](../herramientas/reparar_manifiestos_exportacion.py) | Repara manifiestos históricos con respaldo y comprobación previa de la carga útil; conserva las mallas y sus métricas |

## Etapas

La tabla enlaza el código de cada etapa y resume sus productos. El coordinador combina los argumentos necesarios para ejecutar el flujo completo, que puede iniciarse desde la interfaz o desde el propio coordinador.

| Paso y script | Función y producto principal |
| --- | --- |
| [00_ejecutar_pipeline.py](../procesamiento/00_ejecutar_pipeline.py) | Resuelve referencias, ordena etapas, controla reanudación y almacenamiento |
| [01_crear_mapa_angular.py](../procesamiento/01_crear_mapa_angular.py) | Relaciona capturas y posiciones angulares |
| [02_estimar_profundidad_crestereo.py](../procesamiento/02_estimar_profundidad_crestereo.py) | Rectificación, inferencia ONNX, profundidad, confianza y diagnósticos; mantiene evidencia ROI adicional |
| [03_crear_mascara_objeto.py](../procesamiento/03_crear_mascara_objeto.py) | Separa objeto, fondo y soporte; genera máscaras y diagnósticos del contacto con la base |
| [04_validar_disparidad.py](../procesamiento/04_validar_disparidad.py) | Valida profundidad dentro de la máscara y conserva información de calidad y procedencia |
| [05_crear_consenso_multisesion.py](../procesamiento/05_crear_consenso_multisesion.py) | Compara sesiones y produce profundidad consensuada por pose, con diagnósticos de desacuerdo |
| [06_crear_nubes_puntos.py](../procesamiento/06_crear_nubes_puntos.py) | Proyecta profundidad a nubes métricas con color y atributos de calidad |
| [07_validar_geometria_nubes.py](../procesamiento/07_validar_geometria_nubes.py) | Comprueba geometría de las nubes de referencia para calibración |
| [08_registrar_vistas_referencia.py](../procesamiento/08_registrar_vistas_referencia.py) | Estima registro y parámetros de plataforma utilizando la referencia geométrica |
| [09_guardar_calibracion_plataforma.py](../procesamiento/09_guardar_calibracion_plataforma.py) | Valida y guarda la calibración de plataforma y su auditoría |
| [10_registrar_vistas_calibradas.py](../procesamiento/10_registrar_vistas_calibradas.py) | Aplica la calibración y evalúa ajustes del registro de las vistas |
| [11_fusionar_nubes.py](../procesamiento/11_fusionar_nubes.py) | Integra evidencia multivista mediante modelos locales, incertidumbre y controles de soporte |
| [12_regularizar_nube.py](../procesamiento/12_regularizar_nube.py) | Limpia y regulariza la nube manteniendo sus restricciones de evidencia |
| [13_reconstruir_superficie.py](../procesamiento/13_reconstruir_superficie.py) | Evalúa candidatos de superficie, relleno local y completado de paredes; acepta o descarta según calidad |
| [14_limpiar_topologia.py](../procesamiento/14_limpiar_topologia.py) | Limpieza y reparación topológica con control de intersecciones y bordes |
| [15_pulir_modelo.py](../procesamiento/15_pulir_modelo.py) | Suavizado y pulido controlados; revierte cambios que incumplen las comprobaciones |
| [16_validar_intersecciones.py](../procesamiento/16_validar_intersecciones.py) | Clasifica contactos, intersecciones reales y casos ambiguos |
| [17_validar_modelo.py](../procesamiento/17_validar_modelo.py) | Evalúa fidelidad geométrica, topología y estado final |
| [18_exportar_modelo_blender.py](../procesamiento/18_exportar_modelo_blender.py) | Exporta el modelo validado, versiones métricas, materiales y reporte |

## Módulos compartidos

| Módulo | Uso |
| --- | --- |
| [version_sistema.py](../procesamiento/version_sistema.py) | Versión central del producto utilizada por la interfaz, el coordinador y los registros de calibración, exportación y entorno |
| [clasificador_intersecciones.py](../procesamiento/clasificador_intersecciones.py) | Clasificación geométrica compartida de intersecciones |
| [utilidades_almacenamiento.py](../procesamiento/utilidades_almacenamiento.py) | Conservación y retirada de productos según modo de almacenamiento |
| [utilidades_mascaras.py](../procesamiento/utilidades_mascaras.py) | Operaciones compartidas sobre máscaras |
| [utilidades_multisesion.py](../procesamiento/utilidades_multisesion.py) | Organización y lectura de datos entre sesiones |
| [utilidades_progreso.py](../procesamiento/utilidades_progreso.py) | Comunicación de avance |
| [utilidades_referencias.py](../procesamiento/utilidades_referencias.py) | Copia, resolución y trazabilidad de referencias de campaña |
| [utilidades_calibracion.py](../procesamiento/utilidades_calibracion.py) | Instalación automática desde la aplicación tras aprobar el paso 09, con respaldo y recuperación; revisión independiente opcional por consola |
| [utilidades_estereo.py](../procesamiento/utilidades_estereo.py) | Validación estructural compartida de informes, mapas y matrices estéreo antes de instalar o procesar referencias |
| [utilidades_rendimiento.py](../procesamiento/utilidades_rendimiento.py) | Recursos de CPU, procesos, memoria compartida y límites de concurrencia |
| [utilidades_telemetria.py](../procesamiento/utilidades_telemetria.py) | Medición de tiempos y recursos durante la ejecución |

## Productos de procesamiento

| Ubicación dentro de la campaña | Contenido |
| --- | --- |
| `reconstruccion/S01/02_estimacion_profundidad/` | Profundidad y diagnósticos de la sesión; equivalente en S02 y S03 |
| `reconstruccion/S01/03_mascara_objeto/` | Máscaras y superposiciones de revisión |
| `reconstruccion/S01/04_validacion_disparidad/` | Profundidad validada y revisión por vista |
| `reconstruccion/multisesion/05_consenso_multisesion/` | Consenso y desacuerdo entre sesiones |
| `reconstruccion/multisesion/06_nubes_puntos/` | Nubes por pose |
| `reconstruccion/multisesion/10_registro_calibrado/` | Registro de reconstrucción |
| `reconstruccion/multisesion/11_fusion_multivista/` | Fusión de observaciones |
| `reconstruccion/multisesion/12_regularizacion_nube/` | Nube regularizada |
| `reconstruccion/multisesion/13_reconstruccion_superficie/` | Candidatos, superficie seleccionada y evaluación del relleno |
| `reconstruccion/multisesion/14_limpieza_topologica/` | Limpieza de la malla |
| `reconstruccion/multisesion/15_pulido_final/` | Modelo pulido y controles |
| `reconstruccion/multisesion/16_validacion_intersecciones/` | Evaluación de intersecciones |
| `reconstruccion/multisesion/17_validacion_modelo/` | Validación geométrica final |
| `reconstruccion/multisesion/18_exportacion_modelo/` | Informe de exportación |
| `resultado_final/` | Archivos listos para consulta e importación |

La ruta de calibración usa `07_validacion_geometrica/`, `08_registro_referencia/` y la carpeta de salida indicada para 09. La disponibilidad de intermedios depende del modo de almacenamiento.

## Organización del repositorio

La raíz contiene la aplicación, los lanzadores y la configuración de dependencias. Los scripts numerados de `procesamiento/` representan las etapas de ejecución; las herramientas independientes de `herramientas/` se nombran por su función.

El firmware se guarda en `firmware/control_plataforma_2055_pasos/`, con el mismo nombre para la carpeta y el archivo `.ino`, para abrirlo directamente en Arduino IDE.

`modelos/` contiene los recursos de inferencia y `sistema/` las referencias activas del montaje. Las capturas y resultados permanecen en `trabajos/`, y los registros de ejecución en `registros/`. Estas dos carpetas y los respaldos locales están excluidos de Git. Las referencias de cada campaña se conservan dentro de ella.

La [guía de sistema](../sistema/README.md) describe las referencias globales, sus estados y las condiciones para actualizarlas.

`resultados/` contiene exclusivamente una muestra pública de nueve campañas históricas. Sus archivos y los recursos de `modelos/` y `sistema/` mantienen sus bytes al pasar por Git. Las herramientas de preparación y reparación se ejecutan explícitamente: la aplicación no modifica automáticamente los ejemplos publicados ni los informes históricos.

## Evidencia de la implementación

La [guía de implementación y evidencia](implementacion_y_evidencia.md) relaciona esta arquitectura con las capturas de la interfaz, los controles geométricos y los productos conservados de las etapas.
