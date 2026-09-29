# Modelo de profundidad CREStereo

## Identificación del archivo local

Archivo: `crestereo_init_iter10_480x640.onnx` (25 947 759 bytes).

SHA-256: `ea67a517e61ff381b0b26beec72222f5ba4ef416a251d6e3e1d7224c6f7ba730`.

[identidad_modelo.json](identidad_modelo.json) contiene la inspección realizada con ONNX Runtime: productor `pytorch`, grafo `torch-jit-export`, dos entradas float32 `left` y `right` con forma `[1,3,480,640]`, y salida `output` con forma `[1,2,480,640]`. El campo de versión del grafo no identifica una versión verificable del modelo ni un commit de origen.

## Procedencia y alcance de lo conocido

El responsable identifica el método original como CREStereo de MEGVII Research y declara haber utilizado la conversión ONNX distribuida en [PINTO Model Zoo, 284_CREStereo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo), con el nombre `crestereo_init_iter10_480x640.onnx`. Esta declaración actualiza la referencia de procedencia; no equivale a una comparación binaria con el archivo publicado por el distribuidor.

PINTO ofrece un [script de descarga para la variante iter10](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/284_CREStereo/download_iter10.sh). No se conservan la fecha de adquisición, el commit o paquete exacto descargado, el hash upstream, las versiones del conversor ni el comando de conversión utilizado. No se ha demostrado que este binario sea idéntico a una publicación upstream concreta. La huella anterior identifica exclusivamente el archivo local; el ONNX no se ha sustituido ni reconvertido al actualizar esta documentación.

Referencias relacionadas, consultadas el 26 de septiembre de 2026:

- [Implementación original CREStereo de Megvii](https://github.com/megvii-research/CREStereo), publicada con licencia Apache-2.0.
- [Implementación ONNX de ibaiGorordo](https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation): documenta la conversión desde PyTorch por PINTO0309 y distingue la licencia MIT de sus scripts de Apache-2.0 para los modelos.
- [Conversión PyTorch](https://github.com/ibaiGorordo/CREStereo-Pytorch).
- [PINTO model zoo, 284_CREStereo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo).
- [Artículo CREStereo, CVPR 2022](https://arxiv.org/abs/2203.11483).

Estas referencias explican una cadena pública de implementación y conversión; no acreditan por sí solas la cadena concreta del binario local. El nombre `init_iter10_480x640` sugiere una variante inicial de 10 iteraciones; no se presenta como una versión de conversión verificada.

## Uso en este proyecto

El paso 02 rectifica las imágenes y aplica letterbox manteniendo la relación de aspecto. Convierte BGR a RGB float32, mantiene la escala de intensidades 0–255 y organiza NCHW. La implementación toma el primer canal de salida como disparidad horizontal, retira el padding y recupera la escala de la imagen. La profundidad métrica depende de la calibración estéreo y de la interpretación de la disparidad; el ONNX por sí solo no entrega medidas físicas certificadas.

Las entradas del modelo tienen 640 × 480, aunque la captura y los mapas de rectificación del montaje usan 1920 × 1080. Se ejecuta con ONNX Runtime. `auto` permite fallback; los motores explícitos requieren disponibilidad. La carga local se comprobó con CPU, sin modificar ni reconvertir el modelo.

## Licencia y citación

[LICENSE](LICENSE) es una copia íntegra de la licencia Apache-2.0 del repositorio original CREStereo; [NOTICE.md](NOTICE.md) documenta su alcance y fuente. No es una licencia concedida por el autor de Sistema 3D ni sustituye la comprobación pendiente de la procedencia del binario. No se atribuye MIT al ONNX.

[CITATION.cff](CITATION.cff) y [referencias.bib](referencias.bib) citan el artículo del método. No representan la autoría de la tesis ni una versión certificada del exportador ONNX. La identificación del proyecto propio está en el [README principal](../README.md#identificación-del-proyecto). El código propio de Sistema 3D y su documentación de software se distribuyen bajo la [licencia MIT](../LICENSE); esta licencia no se aplica al ONNX ni reemplaza sus condiciones upstream.

## Limitaciones

La textura escasa, los reflejos, las oclusiones, la iluminación y una rectificación incorrecta pueden producir disparidades erróneas. Los controles LR, de soporte observacional y geométricos del proyecto deben mantenerse. Ninguna aceptación interna demuestra por sí sola exactitud dimensional externa. No cambie este archivo dentro de una campaña congelada: prepare una campaña nueva con las referencias correspondientes.
