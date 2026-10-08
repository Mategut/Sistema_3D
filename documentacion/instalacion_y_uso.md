# Manual de instalación y uso

[Volver al README](../README.md)

Este manual describe la preparación del equipo y el uso de la aplicación. El trabajo se organiza por campañas: cada carpeta conserva las capturas de una adquisición, sus referencias del montaje y sus resultados.

## Cómo utilizar este manual

- **Primera instalación:** siga [Preparar el entorno](#preparar-el-entorno), revise el [montaje](#montaje-de-referencia) y prepare las [calibraciones](#calibraciones-desde-la-interfaz).
- **Reconstruir un objeto:** siga el [recorrido por la interfaz](#reconstruir-un-objeto-paso-a-paso).
- **Continuar un trabajo existente:** consulte [Reanudación](#reanudación).
- **Interpretar el modelo y sus métricas:** consulte [Resultados y validación](resultados_y_validacion.md).
- **Resolver problemas:** consulte [Operación y cierre](operacion_y_cierre.md). Los comandos de consola de este manual son una alternativa al uso de la interfaz.

## Preparar el entorno

Ejecute los comandos desde la raíz del proyecto. Puede utilizar Conda:

```powershell
conda create -n tesis python=3.11
conda activate tesis
python -m pip install -r requirements.txt
python herramientas/verificar_dependencias.py --no-install
```

El verificador revisa NumPy, OpenCV, SciPy, scikit-image, Open3D, Matplotlib, ONNX Runtime, pyserial, Pillow, openpyxl y psutil. PyTorch es opcional. La interfaz utiliza Tkinter, que debe estar disponible en Python.

Instalar `onnxruntime-gpu` no garantiza que CUDA funcione: el controlador y las bibliotecas deben ser compatibles con la versión instalada. Para consultar los proveedores visibles:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Compruebe que `modelos/crestereo_init_iter10_480x640.onnx` existe. La instalación de dependencias no sustituye este archivo.

Los BAT buscan preferentemente el entorno `tesis` en ubicaciones habituales de Conda. Para utilizar el intérprete activo explícitamente:

```powershell
python Sistema_3D.py
```

### Registro del entorno de referencia

[entorno_referencia.json](entorno_referencia.json) identifica Python, el sistema operativo, las distribuciones Python instaladas —incluidas las dependencias transitivas—, el controlador NVIDIA y la huella del ONNX al preparar esta entrega. Describe el entorno de esta entrega. Las campañas históricas pudieron utilizar otra configuración, y el registro no basta para fijar una instalación completa.

El registro no incluye los paquetes exclusivos de Conda, todas las bibliotecas del sistema ni las versiones de CUDA/cuDNN cargadas durante la inferencia. Identificar el controlador NVIDIA tampoco confirma que la inferencia en GPU funcione.

Para registrar otra instalación, ejecute con su intérprete:

```powershell
python herramientas/registrar_entorno.py --output registros/entorno_actual.json
```

La herramienta no instala dependencias ni ejecuta el modelo. `requirements.txt` sigue siendo la entrada de instalación.

## Montaje de referencia

La tabla resume el montaje de referencia y sus dimensiones constructivas aproximadas. El procesamiento utiliza la geometría obtenida de las calibraciones estéreo y de plataforma de cada campaña.

| Elemento | Referencia del montaje |
| --- | --- |
| Cámaras | Dos Logitech Brio 100, sobre una plataforma común rígida, sin convergencia intencional |
| Captura | 1920 × 1080 píxeles; tres sesiones de 25 pares por campaña |
| Separación nominal entre centros ópticos | Aproximadamente 7,7 cm; el informe estéreo incluido estima 77,592 mm |
| Distancia de captura | Aproximadamente 35 cm desde las cámaras al centro; 40 cm desde el borde frontal, con unos 5 cm de retranqueo de cámaras |
| Altura del centro óptico | Aproximadamente 18 cm respecto a la base |
| Estructura | Base de 30 × 50 cm y espesor aproximado de 0,5 cm; giratoria de 18 cm de diámetro y altura de 4–6 cm |
| Cerramiento | Fondo y cubierta de 30 × 30 cm; paneles laterales de 30 × 20 cm; rigidización posterior del soporte de 30 × 10 cm |
| Iluminación | LED interior fijo; superficies uniformes, preferiblemente mates |
| Patrón de calibración | Tablero de 6 × 8 esquinas internas y cuadros de 23 mm; referencia registrada en `calibration_report.json` |
| Control de giro | Arduino Uno; 2055 pasos por vuelta y 25 posiciones |
| Motor paso a paso | 28BYJ-48 unipolar |
| Controlador de potencia | Módulo ULN2003 |
| Alimentación | 5 V del Arduino al terminal positivo de alimentación del ULN2003; masa común entre Arduino y controlador |
| Secuencia de adquisición | Rotar, detener, estabilizar aproximadamente 1 segundo y capturar |

### Parámetros manuales de las cámaras

Ajuste ambas cámaras en el software de Logitech y mantenga esa configuración durante cada sesión. La aplicación de Python solicita la resolución de captura, pero no modifica ni registra esos controles externos.

La configuración de referencia se utiliza con y sin luz natural frontal. Los parámetros fijos evitan cambios automáticos de exposición o color. Aun así, las variaciones de iluminación del entorno pueden afectar las imágenes y la reconstrucción.

Antes de capturar una campaña, ajuste los controles a las condiciones de iluminación y guarde sus valores o una captura de pantalla junto al trabajo. Los resultados publicados no incluyen un registro numérico de estos ajustes. Si cambia la iluminación o la apariencia del fondo, capture de nuevo el fondo vacío.

### Esquemas del montaje

![Vistas general, superior y lateral del montaje estereoscópico](imagenes/esquemas_montaje.png)

Esquema elaborado a partir de las fotografías del montaje real y de sus dimensiones nominales documentadas. Incluye vistas general, superior y lateral; las cotas son aproximadas y el dibujo no está a escala.

### Conexión de Arduino, ULN2003 y motor

![Conexiones del banco de captura y de la plataforma](../firmware/imagenes/conexiones_plataforma.png)

El diagrama y la [guía del firmware](../firmware/README.md) muestran las conexiones funcionales del montaje. Los terminales se identifican mediante sus etiquetas; su posición gráfica no reproduce la distribución física de las placas. Los colores ilustrativos no describen el color real de los cables.

El cableado de control compatible con el firmware incluido utiliza las cuatro entradas del módulo ULN2003:

| Arduino Uno | Módulo ULN2003 |
| --- | --- |
| D8 | IN1 |
| D9 | IN2 |
| D10 | IN3 |
| D11 | IN4 |
| 5 V | Terminal positivo de alimentación (+ / VCC) |
| GND | Terminal negativo de alimentación (− / GND) |

El conector de cinco hilos del 28BYJ-48 se conecta al conector de motor del ULN2003. El motor se acciona a través del controlador, no desde los pines digitales directamente.

El firmware instancia `Stepper(..., 8, 10, 9, 11)`: ese orden corresponde a la secuencia de bobinas IN1, IN3, IN2, IN4. No significa que las conexiones IN2 e IN3 deban intercambiarse. Los pines digitales están verificados en el código; los terminales de alimentación se describen para el montaje a 5 V y no se deducen del firmware.

Cada campaña conserva sus propias referencias de calibración. Para reproducir una adquisición, deben utilizarse las referencias correspondientes al montaje con el que se capturó.

### Equipo de cómputo de referencia

| Componente | Configuración |
| --- | --- |
| Equipo | Computador portátil |
| Sistema operativo | Windows 11 |
| Procesador | AMD Ryzen 7 7435HS |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Memoria RAM | 16 GB DDR5 |

La tabla identifica el equipo de desarrollo. Los informes de cada ejecución registran el proveedor de inferencia utilizado, cuya disponibilidad depende también del entorno instalado. Las dependencias de Python se recogen en `requirements.txt`.

## Plataforma y firmware

El [firmware](../firmware/control_plataforma_2055_pasos/control_plataforma_2055_pasos.ino) utiliza 115200 baudios y los pines de motor 8, 10, 9 y 11. La vuelta calibrada tiene 2055 pasos y 25 movimientos: `[82, 82, 83, 82, 82]` repetidos cinco veces.

| Comando | Función |
| --- | --- |
| `PING` | Comprobar comunicación |
| `STATUS` | Consultar fase y pasos acumulados |
| `NEXT` | Avanzar a la siguiente posición |
| `CLOSE` | Completar el retorno al finalizar la sesión |
| `RESET` | Reiniciar contadores lógicos; no mueve el motor a un origen físico |
| `RELEASE` / `STOP` | Desenergizar el motor sin reiniciar contadores |

Los comandos terminan con salto de línea. El movimiento es bloqueante: `STOP` no interrumpe un avance ya en ejecución. El origen se establece por posición inicial; el montaje no dispone de sensor de referencia.

## Referencias del montaje

### Calibraciones desde la interfaz

La aplicación tiene tres pestañas: **Captura**, **Calibración** y **Herramientas**. El panel lateral permite desplazarse cuando los controles no caben en pantalla.

En **Calibración**:

1. **Crear calibración con tablero** permite indicar esquinas internas, lado del cuadro y separación aproximada de cámaras. Puede capturar imágenes nuevas o calcular con un dataset existente (`left/` y `right/`). Utiliza 1920 × 1080 y los índices de cámara de Captura. Al abrir la captura del tablero se liberan las cámaras de la vista previa; después debe reconectarlas.
2. En la ventana del tablero, **espacio** guarda un par, **Q** termina la captura y calcula, y **R** reinicia los pares de esa captura. **Parar**, en la ventana principal, cancela la operación. Los resultados se guardan en una carpeta nueva dentro de `trabajos/`.
3. **Importar calibración estéreo** instala el resultado seleccionado tras revisar su reporte. Conserva las referencias anteriores en `registros/` e invalida la plataforma anterior. **Verificar estéreo y fondo** comprueba los recursos activos sin modificarlos.
4. **Crear campaña de calibración** prepara la plataforma; continúe con los controles de captura y procesamiento de la pestaña Captura.
5. Al aprobar el paso 09, la aplicación instala automáticamente la calibración y respalda la anterior. **Instalar calibración guardada…** permite instalar un resultado existente con los mismos controles. No se solicita protocolo ni informe adicional: consulte el [flujo de calibración](validacion_independiente_plataforma.md).

En **Herramientas** puede verificar dependencias sin instalar paquetes, auditar archivos `*_lr_state.npy`, consultar el registro y seleccionar motor (`auto`, `cuda`, `directml`, `cpu`) y conservación (`reducido`, `completo`). Los motores explícitos deben estar disponibles. El modo completo conserva los intermedios necesarios para auditorías detalladas. Si ya se compactó el trabajo, cambiar a este modo exige volver a procesar para generar los archivos retirados.

Las herramientas se ejecutan en segundo plano, admiten **Parar** y guardan un registro completo en `registros/`. El visor muestra los últimos 100 000 caracteres. Un error de comprobación aparece como error de operación, con su diagnóstico en el registro. Terminar el cálculo del tablero no significa que su calidad haya sido aceptada: revise `calibration_report.json` antes de importarlo.

| Recurso | Carpeta global | Cuándo revisarlo |
| --- | --- | --- |
| Calibración estéreo | `sistema/calibracion_estereo/` | Cambio de posición relativa o geometría de captura de las cámaras |
| Fondo vacío | `sistema/fondo_vacio/` | Cambio de montaje, fondo o iluminación |
| Calibración de plataforma | `sistema/calibracion_plataforma/` | Cambio de relación entre cámaras y plataforma |

La herramienta de calibración estéreo permite indicar cámaras, resolución, esquinas interiores del tablero (`--cols`, `--rows`) y lado del cuadrado en milímetros (`--square-mm`). Utilice las medidas reales del patrón. `--calibrate-only` procesa imágenes existentes; `--dataset` y `--output` seleccionan las carpetas.

```powershell
python herramientas/calibrar_estereo_checkerboard.py --help
python herramientas/verificar_calibracion_estereo.py
```

Para calibrar la plataforma, utilice un cubo de referencia: los controles geométricos están diseñados para ese objeto. El paso 09 guarda el resultado en `resultado_calibracion_plataforma/` de la campaña y, si aprueba los controles, la aplicación lo instala y respalda la referencia anterior. Todo el procedimiento se completa desde la interfaz, sin un protocolo externo ni un informe del paso 17. Consulte el [flujo de calibración](validacion_independiente_plataforma.md).

## Reconstruir un objeto paso a paso

Antes de comenzar, el modelo ONNX, la calibración estéreo, el fondo vacío y la calibración de plataforma deben corresponder al montaje actual. Si es la primera puesta en marcha o cambió el montaje, complete las [calibraciones](#calibraciones-desde-la-interfaz) antes de crear el objeto.

1. Abra `INICIAR_SISTEMA_3D.bat` o ejecute `python Sistema_3D.py` desde el entorno preparado.
2. En **Captura**, indique el **Puerto** del Arduino, **Baudios** (`115200`) y los índices de **Cám. izq.** y **Cám. der.** Pulse **Conectar cámaras y Arduino**. Revise que ambas vistas correspondan al lado indicado, que el objeto vaya a quedar visible y que el montaje esté fijo. Pulse **Confirmar cámaras listas**.
3. Si necesita actualizar el fondo, retire el objeto y pulse **Capturar fondo vacío**. Hágalo antes de crear la campaña, manteniendo la iluminación y el montaje que usará para capturar.
4. Pulse **Nuevo objeto** e introduzca un nombre corto. La aplicación crea su carpeta dentro de `trabajos/`. Coloque el objeto en la posición inicial y manténgalo fijo respecto a la plataforma durante las tres sesiones.
5. Pulse **Capturar 3 sesiones**. La plataforma gira y la aplicación obtiene 25 pares por sesión. Espere a que termine la captura; no reposicione el objeto entre sesiones. **Continuar** se utiliza únicamente cuando la aplicación solicita un cambio de objeto.
6. En **Herramientas**, seleccione el motor y el almacenamiento. `auto` permite seleccionar un proveedor disponible; `cuda` exige CUDA operativo. `reducido` retira intermedios pesados al finalizar; `completo` los conserva para revisión.
7. Para la primera ejecución, pulse **Procesar desde cero** y espere el resultado de la operación. Para continuar un procesamiento interrumpido, utilice **Reanudar procesamiento**. Procesar desde cero vuelve a generar el procesamiento y sustituye resultados previos de esa campaña.
8. Pulse **Abrir carpeta** y revise `resultado_final/`. Para Blender, utilice `modelo_final.obj` con su archivo de material y siga `README_IMPORTAR_EN_BLENDER.txt`. Para análisis en milímetros, use las versiones identificadas con `mm`. Consulte también los informes de calidad: una operación terminada con advertencias requiere revisar su significado en [Resultados y validación](resultados_y_validacion.md).

Para calibrar la plataforma, use **Calibrar plataforma** en Captura o **Crear campaña de calibración** en Calibración: ambos preparan el mismo tipo de trabajo. Siga el [procedimiento específico](validacion_independiente_plataforma.md) con un cubo como objeto de referencia. Su resultado se guarda en `resultado_calibracion_plataforma/` y se instala automáticamente cuando el paso 09 aprueba sus controles; no se espera una exportación del paso 18.

**Parar** solicita detener la operación actual. Si se interrumpe una captura, revise la posición inicial antes de reiniciarla y lea la confirmación de sustitución de capturas. La reanudación por checkpoints corresponde al procesamiento, no a continuar automáticamente una vuelta física interrumpida.

### Dónde encontrar cada archivo

| Necesidad | Ubicación |
| --- | --- |
| Capturas originales de una campaña | `trabajos/<campaña>/capturas/` |
| Configuración y referencias utilizadas | `trabajos/<campaña>/documentacion/` |
| Informes e intermedios del procesamiento | `trabajos/<campaña>/reconstruccion/` |
| Modelos exportados | `trabajos/<campaña>/resultado_final/` |
| Diagnóstico de una operación | `registros/` y visor de registro de Herramientas |
| Ejemplos publicados | [resultados/](../resultados/README.md) |

Los ejemplos publicados reúnen una selección de productos. Para repetir el procesamiento completo se necesitan todos los pares originales, que no se incluyen en esa selección. Para procesar capturas propias ya existentes, pulse **Reabrir trabajo** y seleccione la carpeta de la campaña dentro de `trabajos/`, no una subcarpeta de imágenes ni la raíz del repositorio.

## Captura

La configuración habitual utiliza tres sesiones (`S01`, `S02`, `S03`), 25 vistas por sesión y resolución de 1920 × 1080. Cada vista contiene una imagen izquierda y otra derecha. Ejemplo de nombre: `OBJ01_PIRAMIDE3_S01_V002_A0144_L.png`.

El ángulo del nombre es nominal; el registro utiliza el mapa angular y la calibración. Mantenga fijos cámaras, iluminación y objeto entre sesiones. Revise enfoque, exposición y visibilidad del contacto con la plataforma antes de completar la campaña.

Las referencias se conservan en `documentacion/referencias/`, con los archivos `configuracion_captura.json` y `referencias_campana.json` en `documentacion/`. Modificar las referencias globales no actualiza automáticamente una campaña existente.

## Ejecución por consola

La interfaz es el punto de entrada habitual. Para ejecutar en PowerShell, adapte campaña e identificador a los datos existentes:

```powershell
$campana = ".\trabajos\NOMBRE_DE_LA_CAMPANA"
$objeto = "IDENTIFICADOR_DEL_OBJETO"
python procesamiento/00_ejecutar_pipeline.py reconstruir `
  --workspace $campana `
  --object $objeto `
  --model ".\modelos\crestereo_init_iter10_480x640.onnx" `
  --stereo-calibration-dir ".\sistema\calibracion_estereo" `
  --background-dir ".\sistema\fondo_vacio" `
  --platform-calibration ".\sistema\calibracion_plataforma\calibracion_plataforma.json" `
  --provider auto `
  --expected-sessions 3 `
  --storage-mode completo `
  --resume
```

`--workspace` es la raíz de **una campaña**, no la del repositorio. El coordinador resuelve las referencias congeladas cuando existen, aunque los argumentos apunten a recursos globales.

| Opción | Comportamiento |
| --- | --- |
| `reconstruir` | Requiere `--platform-calibration` |
| `calibrar-plataforma` | Requiere `--platform-calibration-output-dir` |
| `--provider auto` | Selecciona un proveedor disponible, con alternativa a CPU |
| `--provider cuda` | Solicita CUDA explícitamente; debe estar disponible |
| `--provider cpu` | Inferencia en CPU |
| `--provider directml` | Requiere un entorno que disponga de ese proveedor |
| `--expected-sessions` | Sesiones esperadas; predeterminado: 3 |
| `--storage-mode reducido` | Retira intermedios pesados al finalizar; predeterminado |
| `--storage-mode completo` | Conserva intermedios para revisión |
| `--resume` | Reutiliza checkpoints válidos de la misma campaña |

Para calibración, cambie el modo a `calibrar-plataforma` y sustituya el argumento de calibración de entrada por `--platform-calibration-output-dir "$campana\resultado_calibracion_plataforma"`. La promoción es una operación explícita con la herramienta `promover_calibracion_plataforma.py`, distinta de generar la candidata con el coordinador.

## Reanudación

Los checkpoints de `reconstruccion/estado_pipeline/checkpoints.json` relacionan el código y los argumentos con las entradas y los productos esperados. Al reanudar, el coordinador recalcula las etapas ausentes, rechazadas u obsoletas.

Sin `--resume`, el coordinador prepara una ejecución nueva limpiando resultados de procesamiento. Conserve una copia de la campaña para comparar configuraciones manteniendo ambos resultados.

El modo reducido puede retirar intermedios necesarios para reutilizar etapas. Seleccionar después `completo` no recupera esos archivos: hay que generarlos de nuevo.

## Entorno de referencia

`requirements.txt` reúne las dependencias directas y sus versiones exactas registradas en el entorno de desarrollo Windows con Python 3.11. No es un bloqueo de todas las dependencias transitivas ni sustituye la preparación de CUDA/cuDNN. La instalación automática de paquetes faltantes también utiliza este archivo para respetar esas versiones; no reemplaza por sí sola todos los paquetes ya instalados. El verificador incluye scikit-image: si falta, el paso 13 se detiene en lugar de cambiar silenciosamente el método de superficie.

`auto` permite CPU si no queda disponible aceleración. `cuda` exige una sesión CUDA y bloquea una degradación silenciosa a CPU. Para exigir CUDA usando directamente el paso 02 también existe `--require-cuda`.

Los recursos de `sistema/` y `modelos/` se conservan byte a byte mediante `.gitattributes`; no convierta manualmente sus saltos de línea porque forman parte de las huellas de integridad.

El [recorrido visual y la evidencia de implementación](implementacion_y_evidencia.md) incluye las tres pestañas, fotografías del montaje y diagnósticos conservados.
