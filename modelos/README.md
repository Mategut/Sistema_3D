# Modelo de profundidad CREStereo

## Identificación del archivo local

Archivo: `crestereo_init_iter10_480x640.onnx` (25 947 759 bytes).

SHA-256: `ea67a517e61ff381b0b26beec72222f5ba4ef416a251d6e3e1d7224c6f7ba730`.

[identidad_modelo.json](identidad_modelo.json) contiene la inspección realizada con ONNX Runtime: productor `pytorch`, grafo `torch-jit-export`, dos entradas float32 `left` y `right` con forma `[1,3,480,640]`, y salida `output` con forma `[1,2,480,640]`. El campo de versión del grafo no identifica una versión verificable del modelo ni un commit de origen.

## Procedencia y alcance de lo conocido

El proyecto utiliza CREStereo de MEGVII Research mediante la conversión ONNX distribuida en [PINTO Model Zoo, 284_CREStereo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo), con el nombre `crestereo_init_iter10_480x640.onnx`. La procedencia identifica al distribuidor. No se ha comprobado, mediante comparación binaria, que el archivo local coincida con una descarga concreta de esa fuente.

PINTO ofrece un [script de descarga para la variante iter10](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/284_CREStereo/download_iter10.sh). Esta entrega conserva el ONNX utilizado y lo identifica por su nombre, tamaño y SHA-256.

No se conservaron la fecha de adquisición, el commit o paquete descargado, la huella upstream ni las versiones y el comando de conversión. La información disponible identifica el archivo utilizado, pero no permite comprobar su igualdad binaria con una publicación concreta. El ONNX local se conserva sin sustitución ni reconversión.

Referencias relacionadas, consultadas el 26 de septiembre de 2026:

- [Implementación original CREStereo de Megvii](https://github.com/megvii-research/CREStereo), publicada con licencia Apache-2.0.
- [Implementación ONNX de ibaiGorordo](https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation): documenta la conversión desde PyTorch por PINTO0309 y distingue la licencia MIT de sus scripts de Apache-2.0 para los modelos.
- [Conversión PyTorch](https://github.com/ibaiGorordo/CREStereo-Pytorch).
- [PINTO model zoo, 284_CREStereo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo).
- [Artículo CREStereo, CVPR 2022](https://arxiv.org/abs/2203.11483).

Estas fuentes documentan las implementaciones y conversiones públicas relacionadas con el modelo, aunque no reconstruyen el recorrido exacto del archivo local. El nombre `init_iter10_480x640` sugiere una variante inicial de 10 iteraciones; no se presenta como una versión de conversión verificada.

## Uso en este proyecto

El paso 02 rectifica las imágenes y aplica letterbox manteniendo la relación de aspecto. Convierte BGR a RGB float32, conserva las intensidades en la escala 0–255 y organiza los datos en formato NCHW. La implementación toma el primer canal de salida como disparidad horizontal, retira el padding y recupera la escala de la imagen. Para obtener profundidad métrica se necesitan la calibración estéreo y una interpretación adecuada de la disparidad. La salida del ONNX, por sí sola, no constituye una medida física certificada.

El modelo se ejecuta con ONNX Runtime y recibe imágenes de 640 × 480. Las capturas y los mapas de rectificación del montaje tienen 1920 × 1080. La opción `auto` permite recurrir a otro proveedor disponible; los motores explícitos requieren disponibilidad. La carga local se comprobó con CPU, sin modificar ni reconvertir el modelo.

## Licencia y citación

[LICENSE](LICENSE) es una copia íntegra de la licencia Apache-2.0 del repositorio original CREStereo; [NOTICE.md](NOTICE.md) documenta su alcance y fuente. Las condiciones del modelo corresponden a sus fuentes de terceros; la licencia propia de Sistema 3D no las reemplaza ni acredita identidad binaria upstream. El ONNX no se distribuye bajo la licencia MIT del proyecto.

[CITATION.cff](CITATION.cff) y [referencias.bib](referencias.bib) citan el artículo del método. No representan la autoría de la tesis ni una versión certificada del exportador ONNX. La identificación del proyecto propio está en el [README principal](../README.md#identificación-del-proyecto). El código propio de Sistema 3D y su documentación de software se distribuyen bajo la [licencia MIT](../LICENSE); esta licencia no se aplica al ONNX ni reemplaza sus condiciones upstream.

## Registro del entorno y del binario

El [registro del entorno de referencia](../documentacion/entorno_referencia.json) incorpora las distribuciones Python instaladas y recalcula tamaño y SHA-256 del ONNX local. Puede actualizarse para otra instalación mediante `python herramientas/registrar_entorno.py --output registros/entorno_actual.json`. Este registro permite identificar el archivo y el entorno actuales; no reconstruye la fecha de descarga, el commit upstream ni las herramientas de conversión desconocidas. Las instrucciones y el alcance están en el [manual de instalación](../documentacion/instalacion_y_uso.md#registro-del-entorno-de-referencia).

## Limitaciones

La textura escasa, los reflejos, las oclusiones, la iluminación y una rectificación incorrecta pueden producir disparidades erróneas. Los controles LR, de soporte observacional y geométricos del proyecto deben mantenerse. Ninguna aceptación interna demuestra por sí sola exactitud dimensional externa. No cambie este archivo dentro de una campaña congelada: prepare una campaña nueva con las referencias correspondientes.
