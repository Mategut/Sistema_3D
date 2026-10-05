# Referencia de parámetros declarados

Generada desde los `add_argument` de los scripts. Los valores son los declarados en código; el coordinador puede pasar otros valores explícitos. Para conocer una ejecución use sus informes y checkpoints. `sin valor explícito` no equivale necesariamente a `None`: argparse puede derivar un valor de la acción. Las expresiones se muestran sin ejecutarlas.

Regenerar: `python herramientas/generar_referencia_parametros.py`.

## [00_ejecutar_pipeline.py](../procesamiento/00_ejecutar_pipeline.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `mode` | sin valor explícito | ['calibrar-plataforma', 'reconstruir'] | False | — |
| `--workspace` | sin valor explícito | — | True | Raíz de ESTA campaña/capturas. |
| `--object` | sin valor explícito | — | True | — |
| `--model` | sin valor explícito | — | True | — |
| `--stereo-calibration-dir` | sin valor explícito | — | True | — |
| `--background-dir` | sin valor explícito | — | True | — |
| `--platform-calibration` |  | — | False | Obligatoria para reconstruir. |
| `--platform-calibration-output-dir` |  | — | False | Obligatoria para calibrar-plataforma. |
| `--provider` | auto | ('auto', 'cuda', 'directml', 'cpu') | False | auto usa CUDA si está disponible y cae a DirectML/CPU de forma segura. Los valores explícitos exigen ese provider. |
| `--expected-sessions` | 3 | — | False | — |
| `--regional-workers` | 0 | — | False | argparse.SUPPRESS |
| `--resume` | sin valor explícito | — | False | Reutiliza únicamente checkpoints válidos del MISMO trabajo. Si encuentra un paso ausente, rechazado u obsoleto, continúa desde él. |
| `--storage-mode` | reducido | ('reducido', 'completo') | False | reducido conserva al finalizar solo resúmenes, análisis y previews globales; completo mantiene todos los artefactos intermedios de diagnóstico. |

## [01_crear_mapa_angular.py](../procesamiento/01_crear_mapa_angular.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | cubo | — | False | — |
| `--expected-sessions` | 3 | — | False | — |

## [02_estimar_profundidad_crestereo.py](../procesamiento/02_estimar_profundidad_crestereo.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--model` | sin valor explícito | — | True | Ruta al modelo CREStereo .onnx |
| `--root` | sin valor explícito | — | True | Carpeta raíz Proyecto_3D |
| `--object` | cubo | — | False | Objeto: cubo/cilindro/piramide |
| `--session` | S01 | — | False | Sesión, por ejemplo S01 |
| `--provider` | auto | ('auto', 'cuda', 'directml', 'cpu') | False | — |
| `--require-cuda` | False | — | False | Con provider auto, exige CUDA en lugar de permitir CPU. --provider cuda siempre exige CUDA, incluso con --no-require-cuda. |
| `--limit` | 0 | — | False | Máximo de pares; 0 = todos |
| `--output-name` | 02_estimacion_profundidad | — | False | Nombre de la carpeta de salida |
| `--calibration-dir` |  | — | False | Ruta OBLIGATORIA a la calibración estéreo vigente del sistema. |
| `--clean-output` | True | — | False | Limpia la salida antes de una corrida completa. |
| `--background-dir` |  | — | False | Carpeta con background_left.png y background_right.png. Ruta OBLIGATORIA al fondo vacío vigente del sistema |
| `--background-already-rectified` | sin valor explícito | — | False | — |
| `--background-minimum-threshold` | 12.0 | — | False | — |
| `--background-mad-factor` | 6.0 | — | False | — |
| `--background-minimum-area` | 900 | — | False | — |
| `--foreground-depth-separation-mm` | 18.0 | — | False | Separación mínima respecto al fondo vacío para considerar un píxel como objeto |
| `--geometric-mask-recovery` | True | — | False | Recupera huecos de la máscara RGB únicamente cuando la geometría estéreo y el plano de la plataforma los respaldan. |
| `--geometry-recovery-min-separation-mm` | 6.0 | — | False | Separación mínima por delante del plano local de la plataforma para recuperar un píxel visualmente ausente. |
| `--geometry-recovery-min-support-samples` | 3000 | — | False | Muestras mínimas del anillo exterior de la plataforma para ajustar el plano local. |
| `--geometry-recovery-exclusion-radius-px` | 20 | — | False | Margen excluido alrededor de la semilla RGB al ajustar la plataforma. |
| `--geometry-recovery-max-added-ratio` | 0.75 | — | False | Límite de seguridad: máximo de píxeles recuperados respecto a los píxeles de la semilla RGB. |
| `--minimum-disparity-px` | 0.5 | — | False | — |
| `--lr-abs-tolerance-px` | 1.5 | — | False | — |
| `--lr-relative-tolerance` | 0.02 | — | False | — |
| `--reverse-ratio-minimum` | 0.65 | — | False | — |
| `--reverse-ratio-maximum` | 1.35 | — | False | — |
| `--reverse-median-error-factor` | 2.0 | — | False | La predicción inversa solo se usa cuando su error mediano no supera este múltiplo de la tolerancia mediana |
| `--photo-sigma` | 25.0 | — | False | — |
| `--photo-hard-threshold` | 70.0 | — | False | — |
| `--gradient-sigma` | 65.0 | — | False | — |
| `--smoothness-sigma-px` | 18.0 | — | False | — |
| `--minimum-confidence` | 0.35 | — | False | — |
| `--local-observability-window-px` | 9 | — | False | Ventana local para medir si LR/RL es observable por textura. |
| `--local-observability-minimum-std` | 4.0 | — | False | Desviación estándar local mínima (0..255) exigida en ambas vistas para tratar una discrepancia LR como contradicción fuerte. |
| `--one-sided-minimum-confidence` | 0.5 | — | False | Confianza directa mínima para conservar una observación unilateral débil. |
| `--one-sided-photo-threshold` | 50.0 | — | False | Error fotométrico máximo para una observación unilateral débil. |
| `--one-sided-minimum-smoothness-score` | 0.55 | — | False | Suavidad local mínima para una observación unilateral débil. |
| `--lr-contradiction-factor` | 2.5 | — | False | Una discrepancia LR solo se considera contradicción fuerte cuando supera este múltiplo de la tolerancia y la región es observable. |
| `--minimum-depth-mm` | 150.0 | — | False | — |
| `--maximum-depth-mm` | 1200.0 | — | False | — |
| `--roi-mode` | adaptive | ('adaptive', 'legacy-centered') | False | — |
| `--rect-valid-erosion-px` | 2 | — | False | — |
| `--adaptive-roi-margin-fraction` | 0.18 | — | False | — |
| `--adaptive-roi-minimum-margin-px` | 24 | — | False | — |
| `--adaptive-roi-minimum-component-area` | 900 | — | False | — |
| `--roi-width-fraction` | 0.58 | — | False | — |
| `--roi-height-fraction` | 0.72 | — | False | — |
| `--roi-center-y-fraction` | 0.52 | — | False | — |
| `--minimum-trusted-ratio` | 0.2 | — | False | — |
| `--minimum-center-depth-pixels` | 1000 | — | False | — |
| `--minimum-median-confidence` | 0.45 | — | False | — |
| `--minimum-lower-band-coverage` | 0.55 | — | False | — |
| `--maximum-lower-band-drop` | 0.3 | — | False | — |
| `--epipolar-audit` | True | — | False | Verifica la alineación vertical después de rectificar. Si no puede validarse, el paso se detiene antes de ejecutar CREStereo. |
| `--epipolar-auto-correct` | False | — | False | Compatibilidad experimental. Por defecto NO modifica imágenes ya rectificadas: una discrepancia del auditor se reporta y debe resolverse recalibrando, no deformando la imagen derecha sin actualizar P1/P2/Q. |
| `--epipolar-sift-features` | 12000 | — | False | — |
| `--epipolar-sift-contrast` | 0.005 | — | False | — |
| `--epipolar-match-ratio` | 0.85 | — | False | — |
| `--epipolar-match-max-vertical-px` | 60.0 | — | False | — |
| `--epipolar-ransac-threshold-px` | 1.75 | — | False | — |
| `--epipolar-ransac-iterations` | 6000 | — | False | — |
| `--epipolar-minimum-matches` | 45 | — | False | — |
| `--epipolar-minimum-inliers` | 35 | — | False | — |
| `--epipolar-minimum-inlier-ratio` | 0.25 | — | False | — |
| `--epipolar-minimum-x-span-fraction` | 0.45 | — | False | — |
| `--epipolar-maximum-post-residual-p95-px` | 1.75 | — | False | — |
| `--epipolar-maximum-correction-px` | 8.0 | — | False | — |
| `--epipolar-maximum-vertical-scale-delta` | 0.025 | — | False | — |
| `--epipolar-no-correction-max-px` | 1.5 | — | False | — |
| `--epipolar-audit-strict` | False | — | False | Si está activo, una discrepancia sistemática detectada por SIFT detiene el paso. Por defecto el auditor es diagnóstico: nunca altera P1/P2/Q ni las imágenes rectificadas. |
| `--session-depth-tolerance-mm` | 80.0 | — | False | — |
| `--session-depth-mad-factor` | 6.0 | — | False | — |
| `--session-trusted-ratio-mad-factor` | 4.0 | — | False | — |
| `--keep-rejected-depth` | sin valor explícito | — | False | Conserva profundidad en vistas rechazadas. Solo para diagnóstico. |

## [03_crear_mascara_objeto.py](../procesamiento/03_crear_mascara_objeto.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | cubo | — | False | — |
| `--session` | S01 | — | False | — |
| `--depth-source` | 02_estimacion_profundidad | — | False | Solo se usa esta carpeta para leer la imagen izquierda rectificada, el fondo rectificado y rect_valid_mask. |
| `--output-name` | 03_mascara_objeto | — | False | Se mantiene el nombre histórico para ser drop-in replacement de 03. |
| `--only-angle` | -1.0 | — | False | — |
| `--limit` | 0 | — | False | — |
| `--clean-output` | True | — | False | — |
| `--rect-valid-erosion-px` | 2 | — | False | — |
| `--adaptive-anchor-minimum-threshold` | 7.0 | — | False | — |
| `--adaptive-anchor-mad-factor` | 4.5 | — | False | — |
| `--adaptive-anchor-stable-quantile` | 0.62 | — | False | — |
| `--adaptive-anchor-minimum-component-area` | 900 | — | False | — |
| `--adaptive-anchor-margin-fraction` | 0.18 | — | False | — |
| `--adaptive-anchor-minimum-margin-px` | 24 | — | False | — |
| `--roi-center-x-fraction` | 0.63 | — | False | — |
| `--roi-center-y-fraction` | 0.43 | — | False | — |
| `--roi-width-fraction` | 0.4 | — | False | — |
| `--roi-height-fraction` | 0.76 | — | False | — |
| `--support-cutoff-y-fraction` | 0.8 | — | False | — |
| `--anchor-center-x-fraction` | 0.6 | — | False | — |
| `--anchor-center-y-fraction` | 0.44 | — | False | — |
| `--anchor-width-fraction` | 0.3 | — | False | — |
| `--anchor-height-fraction` | 0.52 | — | False | — |
| `--weak-z-threshold` | 3.3 | — | False | — |
| `--strong-z-threshold` | 5.8 | — | False | — |
| `--minimum-raw-difference` | 8.0 | — | False | — |
| `--chroma-weight` | 1.05 | — | False | — |
| `--luminance-weight` | 0.42 | — | False | — |
| `--gradient-weight` | 0.52 | — | False | — |
| `--shadow-min-darkening-l` | 10.0 | — | False | Oscurecimiento mínimo ΔL (Lab) para considerar sombra. |
| `--shadow-max-chroma-z` | 3.0 | — | False | Sombra: cambio cromático normalizado debe ser bajo. |
| `--shadow-max-gradient-z` | 4.0 | — | False | Sombra: diferencia de gradiente normalizada moderada/baja. |
| `--shadow-strong-override-z` | 8.0 | — | False | Si chroma/gradiente son extremadamente fuertes, no se descarta aunque exista oscurecimiento. |
| `--support-brightening-l` | 6.0 | — | False | — |
| `--support-object-chroma-z` | 4.5 | — | False | — |
| `--support-object-gradient-z` | 6.0 | — | False | — |
| `--seed-anchor-margin-fraction` | 0.22 | — | False | — |
| `--capture-volume-lateral-margin-fraction` | 0.1 | — | False | — |
| `--capture-volume-bottom-margin-fraction` | 0.03 | — | False | — |
| `--support-hardware-exclusion` | True | — | False | — |
| `--support-hardware-lateral-fraction` | 0.028 | — | False | Expansión lateral de la falda respecto al ancho del soporte. |
| `--support-hardware-downward-fraction` | 0.17 | — | False | Expansión hacia abajo respecto a la altura del soporte. |
| `--support-hardware-lower-start-fraction` | 0.4 | — | False | Fracción vertical del bbox del soporte desde la que puede existir la falda; evita expandir el arco superior. |
| `--support-hardware-min-lateral-px` | 6 | — | False | — |
| `--support-hardware-min-downward-px` | 12 | — | False | — |
| `--support-hardware-max-lateral-px` | 64 | — | False | — |
| `--support-hardware-max-downward-px` | 96 | — | False | — |
| `--support-rim-guard` | True | — | False | Excluye un filo fino alrededor del borde físico de la plataforma sin agrandar la superficie útil del soporte. |
| `--support-rim-guard-px` | 6 | — | False | Espesor máximo, en píxeles, del filo exterior de la plataforma. |
| `--support-rim-object-column-margin-px` | 10 | — | False | Margen horizontal alrededor de columnas con cuerpo real del objeto; en esas columnas el filo no se usa como veto. |
| `--support-rim-object-core-clearance-px` | 14 | — | False | Separación mínima respecto del soporte para considerar una semilla como núcleo real del objeto y proteger sus columnas. |
| `--graphcut-recovery` | True | — | False | Recupera zonas ambiguas de baja textura sin borrar foreground ya aceptado. |
| `--graphcut-iterations` | 3 | — | False | — |
| `--graphcut-sure-fg-erosion-px` | 4 | — | False | — |
| `--local-contact-depth-verification` | True | — | False | Usa disparidad contra fondo vacío únicamente dentro de la banda local de contacto. |
| `--local-contact-min-confidence` | 0.045 | — | False | — |
| `--local-contact-max-lr-error-px` | 2.25 | — | False | — |
| `--local-contact-reference-quantile` | 0.52 | — | False | Cuantil de menor cambio RGB usado para aprender el sesgo de disparidad del plato visible. |
| `--local-contact-min-reference-pixels` | 2500 | — | False | — |
| `--local-contact-min-delta-px` | 0.85 | — | False | — |
| `--local-contact-strong-min-delta-px` | 1.6 | — | False | — |
| `--local-contact-mad-factor` | 4.0 | — | False | — |
| `--local-contact-strong-mad-factor` | 6.0 | — | False | — |
| `--local-contact-max-reference-sigma-px` | 1.2 | — | False | Si la disparidad del plato visible es más inestable que esto, la cue estéreo se desactiva para esa vista. |
| `--local-contact-seed-horizontal-radius-px` | 9 | — | False | — |
| `--local-contact-seed-vertical-radius-px` | 34 | — | False | — |
| `--local-contact-close-kernel` | 5 | — | False | — |
| `--local-contact-visual-probability-min` | 0.16 | — | False | — |
| `--local-contact-maximum-growth-iterations` | 400 | — | False | — |
| `--support-occlusion-recovery` | True | — | False | — |
| `--support-occlusion-minimum-raw-difference` | 7.0 | — | False | — |
| `--support-occlusion-strict-mad-factor` | 5.0 | — | False | — |
| `--support-occlusion-loose-mad-factor` | 2.8 | — | False | — |
| `--support-occlusion-stable-quantile` | 0.55 | — | False | — |
| `--support-occlusion-contact-horizontal-radius-px` | 5 | — | False | Tolerancia lateral para conectar objeto con soporte. |
| `--support-occlusion-contact-vertical-radius-px` | 20 | — | False | Tolerancia vertical para conectar objeto con soporte. |
| `--support-occlusion-fallback-radius-px` | 42 | — | False | — |
| `--support-occlusion-minimum-primary-marker-pixels` | 24 | — | False | — |
| `--support-occlusion-minimum-strict-component-area` | 80 | — | False | — |
| `--support-occlusion-minimum-loose-component-area` | 120 | — | False | — |
| `--support-occlusion-open-kernel` | 3 | — | False | — |
| `--support-occlusion-close-kernel` | 5 | — | False | — |
| `--support-occlusion-shadow-dilation-px` | 2 | — | False | — |
| `--support-occlusion-strict-chroma-z` | 3.4 | — | False | — |
| `--support-occlusion-strict-gradient-z` | 5.2 | — | False | — |
| `--support-occlusion-loose-chroma-z` | 1.45 | — | False | — |
| `--support-occlusion-loose-gradient-z` | 2.0 | — | False | — |
| `--support-occlusion-minimum-brightening-l` | 3.0 | — | False | — |
| `--contact-band-horizontal-margin-px` | 10 | — | False | Margen lateral de la banda UNKNOWN en píxeles rectificados. Amplía la zona evaluable sin aceptar automáticamente sus píxeles como objeto. |
| `--contact-band-vertical-gap-px` | 32 | — | False | — |
| `--contact-band-depth-fraction` | 0.9 | — | False | — |
| `--contact-band-min-depth-px` | 55 | — | False | — |
| `--contact-band-max-depth-px` | 320 | — | False | — |
| `--contact-graphcut-iterations` | 5 | — | False | — |
| `--contact-graphcut-sure-fg-erosion-px` | 3 | — | False | — |
| `--contact-connect-horizontal-px` | 7 | — | False | — |
| `--contact-connect-vertical-px` | 18 | — | False | — |
| `--contact-min-component-area` | 120 | — | False | — |
| `--contact-continuity-enabled` | True | — | False | — |
| `--contact-continuity-horizontal-margin-px` | 12 | — | False | — |
| `--contact-continuity-decay-fraction` | 0.58 | — | False | — |
| `--contact-continuity-min-decay-px` | 32 | — | False | — |
| `--contact-continuity-edge-mad-factor` | 3.2 | — | False | — |
| `--contact-continuity-edge-minimum` | 10.0 | — | False | — |
| `--contact-continuity-edge-weight` | 0.75 | — | False | — |
| `--contact-continuity-pr-fg-threshold` | 0.56 | — | False | — |
| `--contact-continuity-rescue-threshold` | 0.68 | — | False | — |
| `--contact-continuity-rescue-appearance-floor` | 0.18 | — | False | — |
| `--open-kernel` | 3 | — | False | — |
| `--close-kernel` | 3 | — | False | — |
| `--maximum-hole-area` | 6000 | — | False | — |
| `--minimum-component-area` | 1200 | — | False | — |
| `--minimum-anchor-overlap-ratio` | 0.015 | — | False | Fracción mínima del componente dentro del anchor. El mejor componente puede superar este filtro por score. |
| `--component-center-sigma-fraction` | 0.22 | — | False | Escala del término de proximidad al centro esperado. |
| `--secondary-component-fraction` | 0.1 | — | False | Componente secundario solo se conserva si es ≥ esta fracción del principal y está espacialmente cerca. |
| `--secondary-max-gap-px` | 22 | — | False | Distancia máxima para unir componentes visuales del objeto. |
| `--minimum-area-ratio` | 0.025 | — | False | — |
| `--maximum-area-ratio` | 0.26 | — | False | — |
| `--session-area-mad-factor` | 6.0 | — | False | — |
| `--session-centroid-tolerance-px` | 120.0 | — | False | — |

## [04_validar_disparidad.py](../procesamiento/04_validar_disparidad.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | cubo | — | False | — |
| `--session` | S01 | — | False | — |
| `--depth-source` | 02_estimacion_profundidad | — | False | — |
| `--silhouette-source` | 03_mascara_objeto | — | False | — |
| `--output-name` | 04_validacion_disparidad | — | False | — |
| `--only-angle` | -1.0 | — | False | — |
| `--limit` | 0 | — | False | — |
| `--calibration-dir` |  | — | False | Ruta opcional a la calibración estéreo del montaje. |
| `--clean-output` | True | — | False | Limpia la salida antes de una corrida completa. |
| `--workers` | 0 | — | False | Procesos paralelos por sesión. 0=automático: usa CPU lógica - 2 (por ejemplo, 16 workers en una CPU de 16 hilos). Usa 1 para ejecución secuencial. |
| `--superpixel-region-size` | 18 | — | False | — |
| `--superpixel-ruler` | 12.0 | — | False | — |
| `--superpixel-iterations` | 12 | — | False | — |
| `--superpixel-min-element-size` | 20 | — | False | — |
| `--seed-minimum-confidence` | 0.62 | — | False | — |
| `--seed-minimum-count` | 24 | — | False | — |
| `--seed-minimum-ratio` | 0.035 | — | False | — |
| `--minimum-disparity-px` | 1.0 | — | False | — |
| `--maximum-disparity-px` | 1500.0 | — | False | — |
| `--minimum-depth-mm` | None | — | False | Límite físico cercano. Si se omite, se hereda de el paso CREStereo. |
| `--maximum-depth-mm` | None | — | False | Límite físico lejano. Si se omite, se hereda de el paso CREStereo. |
| `--ransac-iterations` | 900 | — | False | — |
| `--ransac-threshold-px` | 3.5 | — | False | — |
| `--minimum-inlier-ratio` | 0.62 | — | False | — |
| `--maximum-plane-rmse-px` | 3.2 | — | False | — |
| `--maximum-plane-mad-px` | 2.4 | — | False | — |
| `--minimum-region-area` | 35 | — | False | — |
| `--adaptive-seed-fallback` | True | — | False | Si una región no alcanza suficientes semillas >= seed-minimum-confidence, intenta un segundo conjunto de semillas relativo a la confianza de esa región. |
| `--adaptive-seed-minimum-confidence` | 0.35 | — | False | — |
| `--adaptive-seed-percentile` | 70.0 | — | False | — |
| `--preserve-strict-observations` | True | — | False | Conserva observaciones de alta confianza y rango físico válido aunque el superpíxel no admita un buen modelo plano. |
| `--observed-consensus-fallback` | True | — | False | En regiones sin modelo fiable conserva observaciones REALES de confianza media solo si forman un consenso espacial robusto. Nunca crea disparidad nueva. |
| `--observed-consensus-minimum-confidence` | 0.3 | — | False | — |
| `--observed-consensus-minimum-count` | 12 | — | False | — |
| `--observed-consensus-maximum-mad-px` | 3.5 | — | False | — |
| `--observed-consensus-maximum-rmse-px` | 5.5 | — | False | — |
| `--observed-consensus-sigma` | 3.0 | — | False | — |
| `--observed-consensus-minimum-tolerance-px` | 3.0 | — | False | — |
| `--observed-consensus-maximum-tolerance-px` | 8.0 | — | False | — |
| `--observation-first-validation` | True | — | False | — |
| `--local-consistency-window-px` | 11 | — | False | — |
| `--local-consistency-minimum-neighbors` | 8 | — | False | — |
| `--local-consistency-sigma` | 3.25 | — | False | — |
| `--local-consistency-minimum-tolerance-px` | 3.5 | — | False | — |
| `--local-consistency-maximum-tolerance-px` | 12.0 | — | False | — |
| `--model-adaptive-tolerance-factor` | 2.75 | — | False | — |
| `--model-adaptive-maximum-tolerance-px` | 12.0 | — | False | — |
| `--hard-outlier-model-factor` | 1.85 | — | False | — |
| `--hard-outlier-local-factor` | 1.65 | — | False | — |
| `--extreme-residual-model-factor` | 3.0 | — | False | — |
| `--absolute-hard-outlier-px` | 24.0 | — | False | — |
| `--no-local-support-confidence-floor` | 0.18 | — | False | — |
| `--cross-region-gap-recovery` | True | — | False | Recupera solo huecos SIN observación mediante consenso de al menos dos modelos regionales vecinos compatibles y sin cruzar un borde visual fuerte. |
| `--cross-region-gap-passes` | 2 | — | False | — |
| `--cross-region-minimum-models` | 2 | — | False | — |
| `--cross-region-maximum-model-spread-px` | 6.0 | — | False | — |
| `--cross-region-maximum-boundary-gradient` | 28.0 | — | False | — |
| `--cross-region-minimum-boundary-pixels` | 6 | — | False | — |
| `--validation-tolerance-px` | 6.0 | — | False | — |
| `--fill-accepted-regions` | False | — | False | Completa todo un superpíxel aceptado con el plano regional. Por defecto se conserva únicamente profundidad observada compatible, priorizando datos faltantes frente a geometría inventada. |
| `--observed-blend-minimum` | 0.2 | — | False | — |
| `--observed-blend-maximum` | 0.9 | — | False | — |
| `--model-gap-recovery` | True | — | False | Interpola únicamente huecos sin disparidad dentro de una región con modelo robusto fuerte y cerca de inliers observados. |
| `--model-gap-max-distance-px` | 0.0 | — | False | Distancia máxima a un inlier para completar un hueco. 0 = 1.35 * superpixel-region-size. |
| `--model-gap-minimum-inlier-ratio` | 0.72 | — | False | — |
| `--model-gap-maximum-rmse-px` | 2.6 | — | False | — |
| `--model-gap-maximum-mad-px` | 1.8 | — | False | — |
| `--model-gap-minimum-seeds` | 32 | — | False | — |
| `--model-gap-minimum-seed-confidence-median` | 0.45 | — | False | — |
| `--adjacency-jump-threshold-px` | 10.0 | — | False | — |
| `--adjacency-gradient-threshold` | 30.0 | — | False | — |
| `--adjacency-minimum-boundary-pixels` | 8 | — | False | — |
| `--minimum-valid-mask-ratio` | 0.2 | — | False | — |
| `--maximum-valid-mask-ratio` | 0.98 | — | False | — |
| `--cyclic-half-window` | 2 | — | False | — |
| `--cyclic-minimum-depth-tolerance-mm` | 25.0 | — | False | — |
| `--cyclic-depth-mad-factor` | 6.0 | — | False | — |
| `--closure-depth-tolerance-mm` | 15.0 | — | False | — |
| `--nominal-360-is-closure` | False | — | False | Solo marca A360 como cierre si físicamente coincide con A000. La campaña V7.2 no genera A360; debe permanecer desactivado. |

## [05_crear_consenso_multisesion.py](../procesamiento/05_crear_consenso_multisesion.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | cubo | — | False | — |
| `--manifest` |  | — | False | — |
| `--depth-source` | DEPTH_SOURCE | — | False | — |
| `--silhouette-source` | SILHOUETTE_SOURCE | — | False | — |
| `--regional-source` | REGIONAL_SOURCE | — | False | — |
| `--output-name` | 05_consenso_multisesion | — | False | — |
| `--depth-agreement-mm` | 6.0 | — | False | — |
| `--minimum-independent-depth-support` | 2 | — | False | — |
| `--minimum-valid-ratio` | 0.18 | — | False | — |
| `--adaptive-depth-agreement` | False | — | False | Deriva una escala diagnóstica de repetibilidad. No altera la profundidad ni decide por sí sola si un píxel se exporta. |
| `--depth-agreement-max-mm` | 12.0 | — | False | — |
| `--depth-agreement-mad-factor` | 3.5 | — | False | — |
| `--uncertainty-agreement-sigma` | 2.5 | — | False | Factor de sigma usado para convertir la incertidumbre métrica por píxel en una tolerancia de acuerdo multisesión, siempre acotada entre depth-agreement-mm y depth-agreement-max-mm. |
| `--weak-one-sided-weight` | 0.7 | — | False | Peso relativo de una observación unilateral débil antes de que otra sesión independiente la confirme. |
| `--depth-bias-correction` | False | — | False | Opción histórica conservada por compatibilidad. En V5 los sesgos solo se diagnostican y nunca se aplican al consenso. |
| `--depth-bias-max-mm` | 8.0 | — | False | — |
| `--depth-bias-minimum-pixels` | 1500 | — | False | — |
| `--stable-interior-margin-px` | 6.0 | — | False | Margen mínimo respecto al borde de la silueta para estimar repetibilidad y sesgo Z. |
| `--triplet-interior-rescue` | False | — | False | Opción histórica sin efecto geométrico en V5. Los píxeles con desacuerdo se conservan con confianza explícitamente reducida. |
| `--triplet-rescue-max-span-mm` | 18.0 | — | False | Span corregido máximo para el rescate triplete interior. |
| `--triplet-rescue-warning-ratio` | 0.08 | — | False | Si más de esta fracción de la profundidad válida procede del rescate triplete, la vista queda como warning. |
| `--low-agreement-warning-ratio` | 0.08 | — | False | Fracción máxima de píxeles válidos sin dos observaciones dentro del umbral antes de marcar la pose como warning. |
| `--regional-residual-reference-px` | 3.0 | — | False | Escala de residual regional usada solo para calcular confianza. |
| `--support-depth-veto` | False | — | False | Diagnóstico opcional. Por defecto NO borra píxeles en la zona de contacto objeto-plataforma. |
| `--support-veto-minimum-sessions` | 2 | — | False | — |
| `--support-noise-minimum-mm` | 4.0 | — | False | — |
| `--support-separation-minimum-mm` | 8.0 | — | False | — |
| `--support-separation-maximum-mm` | 24.0 | — | False | — |
| `--support-separation-mad-factor` | 4.0 | — | False | — |
| `--alignment-max-x-px` | 35 | — | False | — |
| `--alignment-max-y-px` | 15 | — | False | — |
| `--alignment-downsample` | 4 | — | False | — |
| `--alignment-local-radius` | 3 | — | False | — |
| `--alignment-fullres-refine-radius` | 4 | — | False | — |
| `--minimum-aligned-silhouette-iou` | 0.72 | — | False | — |

## [06_crear_nubes_puntos.py](../procesamiento/06_crear_nubes_puntos.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | cubo | — | False | — |
| `--consensus-source` | 05_consenso_multisesion | — | False | — |
| `--output-name` | 06_nubes_puntos | — | False | — |
| `--calibration-dir` |  | — | False | — |
| `--pixel-step` | 1 | — | False | — |
| `--voxel-mm` | 0.75 | — | False | — |
| `--sor-neighbors` | 50 | — | False | — |
| `--sor-std-ratio` | 1.55 | — | False | — |
| `--radius-nb-points` | 8 | — | False | — |
| `--radius-mm` | 2.8 | — | False | — |
| `--dbscan-eps-mm` | 4.8 | — | False | — |
| `--dbscan-min-points` | 16 | — | False | — |
| `--minimum-main-cluster-ratio` | 0.58 | — | False | — |
| `--normal-radius-mm` | 5.5 | — | False | — |
| `--normal-max-nn` | 60 | — | False | — |
| `--minimum-final-points` | 4500 | — | False | — |
| `--minimum-point-confidence` | 0.14 | — | False | — |
| `--maximum-depth-spread-mm` | 18.0 | — | False | — |
| `--depth-spread-reference-mm` | 6.0 | — | False | — |
| `--minimum-depth-support` | 2 | — | False | — |
| `--background-depth-source` | 02_estimacion_profundidad | — | False | Subcarpeta de cada sesión que contiene background_depth_mm.npy. Se reutiliza el mismo fondo vacío y se aplican los desplazamientos registrados por el paso 05. |
| `--foreground-weak-z-score` | 1.5 | — | False | — |
| `--foreground-strong-z-score` | 4.0 | — | False | — |
| `--background-noise-floor-mm` | 2.0 | — | False | — |
| `--background-relative-noise` | 0.003 | — | False | — |
| `--background-minimum-models` | 1 | — | False | — |
| `--foreground-component-minimum-strong-pixels` | 4 | — | False | — |
| `--foreground-boundary-recovery-px` | 2.0 | — | False | — |
| `--component-minimum-foreground-score` | 0.55 | — | False | — |
| `--disparity-uncertainty-floor-px` | 0.35 | — | False | — |
| `--stereo-weak-score` | 0.34 | — | False | — |
| `--stereo-strong-score` | 0.62 | — | False | — |
| `--stereo-minimum-observed-sessions` | 2 | — | False | — |
| `--stereo-lr-sigma-factor` | 3.0 | — | False | — |
| `--stereo-regional-sigma-factor` | 3.0 | — | False | — |
| `--stereo-local-neighbor-z-score` | 3.5 | — | False | — |
| `--stereo-minimum-consistent-neighbors` | 2 | — | False | — |
| `--stereo-gradient-edge-relaxation` | 5.0 | — | False | — |
| `--secondary-cluster-minimum-ratio` | 0.02 | — | False | — |
| `--secondary-cluster-maximum-distance-mm` | 0.0 | — | False | Límite opcional entre componentes. Cero lo desactiva para no imponer un tamaño máximo al objeto. |
| `--secondary-cluster-minimum-confidence` | 0.38 | — | False | — |
| `--inverse-depth-regularization` | True | — | False | — |
| `--inverse-depth-iterations` | 8 | — | False | — |
| `--inverse-depth-smoothness` | 1.35 | — | False | — |
| `--inverse-depth-data-strength` | 1.0 | — | False | — |
| `--inverse-depth-edge-sigma-mm` | 2.5 | — | False | — |
| `--inverse-depth-color-sigma` | 0.12 | — | False | — |
| `--inverse-depth-maximum-displacement-mm` | 1.8 | — | False | — |
| `--surface-smoothing` | True | — | False | — |
| `--surface-smoothing-iterations` | 1 | — | False | — |
| `--surface-smoothing-radius-mm` | 3.2 | — | False | — |
| `--surface-smoothing-max-neighbors` | 44 | — | False | — |
| `--surface-smoothing-min-neighbors` | 6 | — | False | — |
| `--surface-smoothing-normal-angle-deg` | 32.0 | — | False | — |
| `--surface-smoothing-plane-sigma-mm` | 0.85 | — | False | — |
| `--surface-smoothing-strength` | 0.3 | — | False | — |
| `--surface-smoothing-max-step-mm` | 0.2 | — | False | — |
| `--local-residual-reference-mm` | 0.9 | — | False | — |

## [07_validar_geometria_nubes.py](../procesamiento/07_validar_geometria_nubes.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | cubo | — | False | — |
| `--session` | S01 | — | False | — |
| `--cloud-source` | 06_nubes_puntos | — | False | — |
| `--output-name` | 07_validacion_geometrica | — | False | — |
| `--only-angle` | -1.0 | — | False | — |
| `--limit` | 0 | — | False | — |
| `--clean-output` | True | — | False | — |
| `--seed` | 500 | — | False | — |
| `--maximum-planes` | 3 | — | False | — |
| `--ransac-iterations` | 1200 | — | False | — |
| `--ransac-evaluation-points` | 9000 | — | False | — |
| `--plane-distance-threshold-mm` | 3.5 | — | False | — |
| `--minimum-plane-points` | 1200 | — | False | — |
| `--minimum-plane-ratio-total` | 0.045 | — | False | — |
| `--minimum-plane-ratio-remaining` | 0.1 | — | False | — |
| `--refinement-rounds` | 4 | — | False | — |
| `--minimum-cloud-points` | 4500 | — | False | — |
| `--minimum-best-plane-ratio` | 0.18 | — | False | — |
| `--minimum-explained-ratio` | 0.55 | — | False | — |
| `--warning-plane-rmse-mm` | 2.6 | — | False | — |
| `--warning-plane-p95-mm` | 4.5 | — | False | — |
| `--warning-plane-thickness95-mm` | 7.0 | — | False | — |
| `--warning-normal-p95-deg` | 32.0 | — | False | — |
| `--rejection-best-plane-rmse-mm` | 4.5 | — | False | — |
| `--parallel-angle-deg` | 12.0 | — | False | — |
| `--layer-minimum-separation-mm` | 4.0 | — | False | — |
| `--layer-minimum-plane-ratio` | 0.07 | — | False | — |
| `--geometry-mode` | auto | ('auto', 'generic', 'cuboid') | False | — |
| `--cuboid-minimum-orthogonal-angle-deg` | 72.0 | — | False | — |
| `--cuboid-major-plane-ratio` | 0.1 | — | False | — |
| `--preview-width` | 620 | — | False | — |
| `--preview-height` | 450 | — | False | — |

## [08_registrar_vistas_referencia.py](../procesamiento/08_registrar_vistas_referencia.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | cubo | — | False | — |
| `--session` | multisesion | — | False | — |
| `--cloud-source` | 06_nubes_puntos | — | False | — |
| `--geometry-source` | 07_validacion_geometrica | — | False | — |
| `--stereo-calibration-dir` | sin valor explícito | — | True | Calibración estéreo vigente. La baseline del montaje se deriva de T/P2/Q; no se acepta una constante histórica. |
| `--output-name` | 08_registro_referencia | — | False | — |
| `--manifest` |  | — | False | — |
| `--clean-output` | True | — | False | — |
| `--seed` | 500 | — | False | — |
| `--expected-views` | 25 | — | False | — |
| `--expected-steps-per-revolution` | 2055 | — | False | — |
| `--angle-validation-tolerance-deg` | 0.02 | — | False | — |
| `--minimum-registration-points` | 3500 | — | False | — |
| `--pose-plane-minimum-ratio` | 0.055 | — | False | — |
| `--pose-plane-hard-rmse-mm` | 3.0 | — | False | — |
| `--oblique-minor-ratio-to-major` | 0.58 | — | False | — |
| `--oblique-absolute-minor-ratio` | 0.2 | — | False | — |
| `--very-dispersed-normal-p95-deg` | 70.0 | — | False | — |
| `--very-dispersed-max-ratio` | 0.14 | — | False | — |
| `--axis-plane-minimum-ratio` | 0.1 | — | False | — |
| `--axis-pair-minimum-angle-deg` | 72.0 | — | False | — |
| `--axis-pair-maximum-angle-deg` | 108.0 | — | False | — |
| `--axis-consensus-tolerance-deg` | 9.0 | — | False | — |
| `--axis-minimum-support-views` | 4 | — | False | — |
| `--axis-camera-vertical-max-deg` | 38.0 | — | False | Escala angular del prior SUAVE hacia Y de cámara. No limita el eje durante la calibración salvo que se active --axis-use-camera-vertical-hard-gate. |
| `--axis-use-camera-vertical-hard-gate` | False | — | False | Compatibilidad experimental. Por defecto False porque la orientación global del rig puede cambiar (por ejemplo, cámaras inclinadas hacia abajo). |
| `--axis-ba-limit-deg` | 2.0 | — | False | — |
| `--axis-prior-sigma-deg` | 0.9 | — | False | — |
| `--axis-multiview-plane-minimum-ratio` | 0.07 | — | False | — |
| `--axis-multiview-search-limit-deg` | 6.0 | — | False | — |
| `--axis-multiview-prior-sigma-deg` | 3.0 | — | False | — |
| `--axis-multiview-huber-deg` | 8.0 | — | False | — |
| `--axis-multiview-maxiter` | 28 | — | False | — |
| `--axis-multiview-popsize` | 8 | — | False | — |
| `--axis-fallback-enabled` | True | — | False | Si el consenso estricto de pares ortogonales falla, estima una semilla mediante coherencia Manhattan multivista y ángulos físicos. |
| `--axis-fallback-relaxed-min-angle-deg` | 45.0 | — | False | — |
| `--axis-fallback-relaxed-max-angle-deg` | 135.0 | — | False | — |
| `--axis-fallback-max-seeds` | 8 | — | False | — |
| `--axis-fallback-seed-separation-deg` | 2.5 | — | False | — |
| `--axis-fallback-search-limit-deg` | 10.0 | — | False | — |
| `--axis-fallback-seed-prior-sigma-deg` | 6.0 | — | False | — |
| `--axis-fallback-baseline-weight` | 0.18 | — | False | — |
| `--axis-cloud-fallback-enabled` | True | — | False | Cuando 08 entra por el fallback de eje, refina conjuntamente eje, centro y sentido usando las nubes actuales y los ángulos físicos. |
| `--axis-cloud-search-limit-deg` | 8.0 | — | False | — |
| `--axis-cloud-center-radius-mm` | 95.0 | — | False | — |
| `--axis-cloud-maxiter` | 11 | — | False | — |
| `--axis-cloud-popsize` | 5 | — | False | — |
| `--axis-cloud-tail-weight` | 0.35 | — | False | — |
| `--axis-cloud-secondary-weight` | 0.35 | — | False | — |
| `--axis-cloud-mount-multiplier` | 1.8 | — | False | — |
| `--axis-cloud-overlap-target` | 0.55 | — | False | — |
| `--axis-cloud-overlap-penalty-mm` | 7.0 | — | False | — |
| `--fallback-ba-tangential-weight` | 0.22 | — | False | — |
| `--fallback-ba-mount-prior-equivalent-points` | 360.0 | — | False | — |
| `--fallback-ba-center-prior-equivalent-points` | 90.0 | — | False | — |
| `--auto-stabilize-low-observability` | True | — | False | — |
| `--auto-stabilize-low-ratio` | 0.5 | — | False | Activa BA estabilizado si esta fracción de poses es low. |
| `--auto-ba-tangential-weight` | 0.22 | — | False | — |
| `--auto-ba-mount-prior-equivalent-points` | 300.0 | — | False | — |
| `--auto-ba-center-prior-equivalent-points` | 75.0 | — | False | — |
| `--stereo-baseline-mm` | None | — | False | Compatibilidad: si se proporciona, debe coincidir con stereo_initial.yaml. |
| `--mount-midpoint-axis-distance-mm` | 350.0 | — | False | Distancia nominal desde el punto medio estereo al eje: 400 mm desde el borde menos 50 mm de retranqueo de camaras. |
| `--mount-distance-sigma-mm` | 30.0 | — | False | — |
| `--mount-symmetry-sigma-mm` | 25.0 | — | False | — |
| `--mount-axis-baseline-sigma-deg` | 6.0 | — | False | — |
| `--mount-coarse-prior-weight-mm` | 1.2 | — | False | — |
| `--mount-prior-equivalent-points` | 180.0 | — | False | — |
| `--axis-prior-equivalent-points` | 90.0 | — | False | — |
| `--center-prior-equivalent-points` | 45.0 | — | False | — |
| `--angle-prior-equivalent-points` | 8.0 | — | False | — |
| `--angle-smoothness-equivalent-points` | 9.0 | — | False | — |
| `--angle-curvature-equivalent-points` | 7.0 | — | False | — |
| `--maximum-edge-order` | 2 | — | False | — |
| `--maximum-edge-gap-deg` | 30.5 | — | False | — |
| `--secondary-edge-weight` | 0.72 | — | False | — |
| `--optimization-voxel-mm` | 2.0 | — | False | — |
| `--coarse-max-points-per-view` | 900 | — | False | — |
| `--fine-max-points-per-view` | 3200 | — | False | — |
| `--metric-trim-fraction` | 0.72 | — | False | — |
| `--metric-overlap-distance-mm` | 5.0 | — | False | — |
| `--metric-distance-cap-mm` | 15.0 | — | False | — |
| `--metric-overlap-penalty-mm` | 5.0 | — | False | — |
| `--center-global-search-radius-mm` | 75.0 | — | False | — |
| `--center-global-maxiter` | 18 | — | False | — |
| `--center-global-popsize` | 8 | — | False | — |
| `--center-local-radius-mm` | 28.0 | — | False | — |
| `--center-ba-limit-mm` | 30.0 | — | False | — |
| `--center-prior-sigma-mm` | 24.0 | — | False | — |
| `--ba-rounds` | 4 | — | False | — |
| `--ba-geometry-only-rounds` | 2 | — | False | Primeras rondas donde las correcciones angulares quedan prácticamente congeladas para resolver primero eje/centro. |
| `--ba-geometry-only-angle-limit-deg` | 0.02 | — | False | — |
| `--ba-geometry-only-angle-sigma-deg` | 0.02 | — | False | — |
| `--ba-correspondence-gates-mm` | 7.0,5.5,4.5,3.8 | — | False | — |
| `--ba-max-correspondences-per-direction` | 320 | — | False | — |
| `--ba-min-correspondences-per-direction` | 55 | — | False | — |
| `--ba-point-plane-sigma-mm` | 1.8 | — | False | — |
| `--ba-tangential-weight` | 0.1 | — | False | — |
| `--ba-tangential-sigma-mm` | 5.0 | — | False | — |
| `--ba-robust-loss` | soft_l1 | ('linear', 'soft_l1', 'huber', 'cauchy') | False | — |
| `--ba-f-scale` | 1.0 | — | False | — |
| `--ba-max-nfev` | 55 | — | False | — |
| `--angle-correction-limit-deg` | 0.6 | — | False | — |
| `--angle-prior-sigma-deg` | 0.24 | — | False | — |
| `--angle-smoothness-sigma-deg` | 0.32 | — | False | — |
| `--angle-curvature-sigma-deg` | 0.22 | — | False | — |
| `--angular-plane-minimum-ratio` | 0.055 | — | False | — |
| `--angular-minimum-tangent-sensitivity` | 0.45 | — | False | — |
| `--angular-minimum-second-plane-weight` | 0.1 | — | False | — |
| `--angular-full-diversity-second-weight` | 0.28 | — | False | — |
| `--angular-support-reference` | 0.75 | — | False | — |
| `--angular-observability-power` | 1.35 | — | False | — |
| `--angular-low-limit-deg` | 0.18 | — | False | — |
| `--angular-high-limit-deg` | 0.6 | — | False | — |
| `--angular-low-sigma-deg` | 0.085 | — | False | — |
| `--angular-high-sigma-deg` | 0.24 | — | False | — |
| `--angular-normal-prior-max-target-deg` | 0.1 | — | False | — |
| `--angular-normal-prior-scatter-sigma-deg` | 0.85 | — | False | — |
| `--angular-normal-prior-mean-sigma-deg` | 2.5 | — | False | — |
| `--angular-normal-prior-min-reliability` | 0.5 | — | False | — |
| `--angular-low-prior-weight` | 1.45 | — | False | — |
| `--angular-high-prior-weight` | 0.8 | — | False | — |
| `--history-overlap-reference` | 0.55 | — | False | — |
| `--history-rmse-reference-mm` | 3.0 | — | False | — |
| `--history-minimum-weight` | 0.2 | — | False | — |
| `--final-overlap-distance-mm` | 4.5 | — | False | — |
| `--final-primary-minimum-overlap` | 0.4 | — | False | — |
| `--final-primary-maximum-rmse-mm` | 3.3 | — | False | — |
| `--final-primary-maximum-p90-mm` | 7.5 | — | False | — |
| `--warning-primary-acceptance` | 0.72 | — | False | — |
| `--reject-primary-acceptance` | 0.55 | — | False | — |
| `--warning-median-primary-overlap` | 0.55 | — | False | — |
| `--reject-median-primary-overlap` | 0.35 | — | False | — |
| `--warning-median-primary-rmse-mm` | 2.8 | — | False | — |
| `--reject-median-primary-rmse-mm` | 4.0 | — | False | — |
| `--warning-closure-rmse-mm` | 3.3 | — | False | — |
| `--reject-closure-rmse-mm` | 5.0 | — | False | — |
| `--final-primary-maximum-point-plane-rmse-mm` | 2.4 | — | False | — |
| `--final-primary-maximum-point-plane-p90-mm` | 3.8 | — | False | — |
| `--tangential-warning-p90-mm` | 10.0 | — | False | P90 euclídeo alto se reporta como warning tangencial, no como fallo de superficie. |
| `--warning-mount-distance-error-mm` | 35.0 | — | False | — |
| `--reject-mount-distance-error-mm` | 65.0 | — | False | — |
| `--warning-mount-symmetry-error-mm` | 25.0 | — | False | — |
| `--reject-mount-symmetry-error-mm` | 50.0 | — | False | — |
| `--warning-axis-baseline-deviation-deg` | 7.0 | — | False | — |
| `--reject-axis-baseline-deviation-deg` | 14.0 | — | False | — |
| `--compactness-normal-gate-deg` | 25.0 | — | False | — |
| `--compactness-minimum-points-per-face` | 500 | — | False | — |
| `--compactness-min-face-separation-mm` | 18.0 | — | False | Separación mínima para interpretar dos clusters 1D como caras opuestas y no como una sola cara gruesa. |
| `--warning-global-face-p90-mm` | 5.5 | — | False | — |
| `--reject-global-face-p90-mm` | 9.0 | — | False | — |
| `--warning-manhattan-p90-deg` | 14.0 | — | False | — |
| `--reject-manhattan-p90-deg` | 24.0 | — | False | — |
| `--warning-max-angle-correction-deg` | 0.55 | — | False | — |
| `--reject-max-angle-correction-deg` | 0.74 | — | False | — |
| `--angular-saturation-ratio` | 0.985 | — | False | — |
| `--warning-total-angular-saturations` | 3 | — | False | — |
| `--reject-total-angular-saturations` | 9 | — | False | — |
| `--warning-high-observability-saturations` | 1 | — | False | — |
| `--reject-high-observability-saturations` | 5 | — | False | — |
| `--high-observability-threshold` | 0.65 | — | False | — |
| `--warning-angular-prior-residual-deg` | 0.4 | — | False | — |
| `--reject-angular-prior-residual-deg` | 0.8 | — | False | — |
| `--warning-union-extent-ratio` | 1.75 | — | False | — |
| `--reject-union-extent-ratio` | 2.4 | — | False | — |
| `--union-voxel-mm` | 0.85 | — | False | — |
| `--preview-width` | 760 | — | False | — |
| `--preview-height` | 520 | — | False | — |

## [09_guardar_calibracion_plataforma.py](../procesamiento/09_guardar_calibracion_plataforma.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--source-object` | cubo | — | False | — |
| `--registration-source` | 08_registro_referencia | — | False | — |
| `--output-dir` | sin valor explícito | — | True | Directorio persistente donde guardar la calibración de plataforma. |
| `--stereo-calibration-dir` | sin valor explícito | — | True | Calibración estéreo vigente; no se buscan calibraciones históricas. |
| `--baseline-mm` | None | — | False | Compatibilidad manual únicamente. La baseline se lee siempre de stereo_initial.yaml; si se proporciona este valor debe coincidir. |
| `--mount-reference-distance-mm` | 350.0 | — | False | Distancia nominal desde el punto medio estereo al eje: 400 mm desde el borde menos 50 mm de retranqueo de camaras. |
| `--mount-reference-sigma-mm` | 30.0 | — | False | — |
| `--minimum-primary-surface-accepted-ratio` | 0.95 | — | False | — |
| `--maximum-primary-point-plane-rmse-median-mm` | 2.5 | — | False | — |
| `--maximum-primary-point-plane-p90-median-mm` | 4.0 | — | False | — |
| `--minimum-closure-overlap` | 0.8 | — | False | — |
| `--maximum-closure-rmse-mm` | 2.5 | — | False | — |
| `--maximum-closure-point-plane-rmse-mm` | 2.0 | — | False | — |
| `--maximum-angle-correction-deg` | 0.35 | — | False | — |
| `--maximum-adaptive-saturations` | 1 | — | False | — |
| `--maximum-manhattan-p90-deg` | 12.0 | — | False | — |
| `--maximum-reference-compactness-p90-mm` | 6.0 | — | False | — |
| `--maximum-mount-reference-error-mm` | 65.0 | — | False | — |

## [10_registrar_vistas_calibradas.py](../procesamiento/10_registrar_vistas_calibradas.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | sin valor explícito | — | True | — |
| `--calibration` | sin valor explícito | — | True | Calibración de plataforma vigente. Nunca se buscan resultados previos. |
| `--stereo-calibration-dir` | sin valor explícito | — | True | Calibración estéreo vigente. Debe coincidir exactamente con el contrato con el que se generó la calibración de plataforma. |
| `--cloud-source` | 06_nubes_puntos | — | False | — |
| `--output-name` | 10_registro_calibrado | — | False | — |
| `--expected-views` | 25 | — | False | — |
| `--sample-points-per-view` | 12000 | — | False | — |
| `--seed` | 500 | — | False | — |
| `--overlap-distance-mm` | 4.5 | — | False | — |
| `--metric-cap-mm` | 15.0 | — | False | — |
| `--minimum-informative-overlap` | 0.12 | — | False | — |
| `--surface-maximum-point-plane-rmse-mm` | 2.8 | — | False | — |
| `--surface-maximum-point-plane-p90-mm` | 4.5 | — | False | — |
| `--minimum-informative-primary-ratio` | 0.6 | — | False | — |
| `--minimum-accepted-informative-primary-ratio` | 0.85 | — | False | — |
| `--closure-minimum-overlap` | 0.2 | — | False | — |
| `--closure-maximum-point-plane-rmse-mm` | 2.8 | — | False | — |
| `--closure-maximum-point-plane-p90-mm` | 4.5 | — | False | — |
| `--union-voxel-mm` | 0.8 | — | False | — |
| `--minimum-point-confidence` | 0.2 | — | False | — |
| `--maximum-point-spread-mm` | 12.0 | — | False | — |
| `--point-spread-reference-mm` | 6.0 | — | False | — |
| `--minimum-reliable-points-per-view` | 2200 | — | False | — |
| `--confidence-sampling-power` | 1.5 | — | False | — |
| `--reprojection-neighbor-pose-radius` | 3 | — | False | — |
| `--reprojection-pixel-radius` | 2.5 | — | False | — |
| `--reprojection-depth-z-score` | 3.5 | — | False | — |
| `--reprojection-noise-floor-mm` | 1.0 | — | False | — |
| `--reprojection-minimum-support-poses` | 2 | — | False | — |
| `--reprojection-minimum-independent-agreements` | 2 | — | False | — |
| `--reprojection-minimum-agreement-ratio` | 0.7 | — | False | — |
| `--reprojection-minimum-angular-span-poses` | 2 | — | False | — |
| `--reprojection-maximum-contradictions` | 1 | — | False | — |
| `--reprojection-front-facing-cosine` | 0.05 | — | False | — |
| `--weak-boundary-minimum-foreground-score` | 0.45 | — | False | — |
| `--weak-boundary-recovery-hops` | 2 | — | False | — |
| `--minimum-clean-union-voxels` | 500 | — | False | — |
| `--silhouette-boundary-tolerance-px` | 4.0 | — | False | — |
| `--silhouette-minimum-tested-poses` | 12 | — | False | — |
| `--silhouette-strong-inside-ratio` | 0.88 | — | False | — |
| `--silhouette-weak-inside-ratio` | 0.76 | — | False | — |
| `--silhouette-maximum-strong-contradictions` | 3 | — | False | — |
| `--silhouette-maximum-weak-contradictions` | 6 | — | False | — |
| `--static-hypothesis-neighbor-radius` | 4 | — | False | — |
| `--static-hypothesis-gate-mm` | 2.4 | — | False | — |
| `--static-hypothesis-minimum-matches` | 3 | — | False | — |
| `--static-hypothesis-dominance-margin` | 2 | — | False | — |
| `--static-hypothesis-maximum-removal-ratio` | 0.3 | — | False | — |
| `--refine-axis-line` | False | — | False | Diagnóstico opcional: refina la línea del eje con el objeto. Desactivado por defecto para mantener poses congeladas. |
| `--apply-axis-line-refinement` | False | — | False | Autoriza aplicar el candidato refinado si supera todas las guardas A/B. Por defecto solo se evalúa y se conserva la calibración congelada. |
| `--axis-line-search-limit-mm` | 12.0 | — | False | — |
| `--axis-line-optimization-points` | 1800 | — | False | — |
| `--axis-line-trim-axial-percent` | 12.0 | — | False | — |
| `--axis-line-trim-transverse-percent` | 2.0 | — | False | — |
| `--axis-line-minimum-improvement-ratio` | 0.06 | — | False | Parámetro legacy conservado por compatibilidad; V3.2 usa los gates proposal/reserved/full. |
| `--axis-line-proposal-minimum-improvement-ratio` | 0.005 | — | False | — |
| `--axis-line-reserved-minimum-improvement-ratio` | 0.005 | — | False | — |
| `--axis-line-maximum-split-disagreement-mm` | 3.5 | — | False | — |
| `--axis-line-uncertainty-offset-factor` | 1.5 | — | False | Límite inicial del desplazamiento como múltiplo de la incertidumbre mediana de profundidad; una ganancia independiente demostrada puede ampliarlo de forma acotada. |
| `--axis-line-maximum-uncertainty-offset-mm` | 4.0 | — | False | — |
| `--axis-line-conservative-gain-fraction` | 0.82 | — | False | Fracción mínima de la mejor ganancia observada que debe conservar un candidato. Se mantiene por compatibilidad con ejecuciones previas. |
| `--axis-line-minimum-closure-improvement-ratio` | 0.0 | — | False | Mejora mínima exigida en el cierre para aceptar un candidato. |
| `--axis-line-cross-validation-balance-weight` | 0.1 | — | False | Penalización por desequilibrio entre las validaciones par e impar. |
| `--axis-line-full-minimum-rmse-improvement-ratio` | 0.02 | — | False | — |
| `--axis-line-full-minimum-p90-improvement-ratio` | 0.01 | — | False | — |
| `--axis-line-maximum-consecutive-rmse-degradation-ratio` | 0.015 | — | False | — |
| `--axis-line-maximum-consecutive-p90-degradation-ratio` | 0.015 | — | False | — |
| `--axis-line-maximum-consecutive-absolute-degradation-mm` | 0.03 | — | False | — |
| `--axis-line-maximum-consecutive-overlap-drop` | 0.015 | — | False | — |
| `--axis-line-maximum-degraded-consecutive-edges` | 0 | — | False | — |
| `--axis-line-regional-bands` | 3 | — | False | — |
| `--axis-line-regional-minimum-points` | 250 | — | False | — |
| `--axis-line-regional-maximum-rmse-degradation-ratio` | 0.02 | — | False | — |
| `--axis-line-regional-maximum-p90-degradation-ratio` | 0.02 | — | False | — |
| `--axis-line-regional-maximum-absolute-degradation-mm` | 0.05 | — | False | — |
| `--axis-line-regional-maximum-overlap-drop` | 0.02 | — | False | — |
| `--axis-line-maximum-degraded-regions` | 0 | — | False | — |
| `--registration-warning-rmse-uncertainty-ratio` | 0.6 | — | False | — |
| `--registration-warning-p90-uncertainty-ratio` | 0.95 | — | False | — |
| `--registration-warning-closure-overlap` | 0.85 | — | False | — |
| `--registration-reject-closure-overlap` | 0.7 | — | False | Gate duro para P24->P00 cuando el cierre tiene correspondencias informativas. Una vuelta mecánica no puede terminar mucho peor que sus pares consecutivos. |
| `--registration-min-closure-to-primary-overlap-ratio` | 0.75 | — | False | Relación mínima entre overlap de cierre y mediana de overlaps primarios informativos. Es agnóstica a la forma porque compara el objeto consigo mismo. |
| `--registration-warning-input-confidence-median` | 0.72 | — | False | — |
| `--axis-line-maximum-evaluations` | 220 | — | False | — |

## [11_fusionar_nubes.py](../procesamiento/11_fusionar_nubes.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | sin valor explícito | — | True | — |
| `--calibration` | sin valor explícito | — | True | Calibración de plataforma vigente. |
| `--cloud-source` | 06_nubes_puntos | — | False | — |
| `--consensus-source` | 05_consenso_multisesion | — | False | — |
| `--registration-source` | 10_registro_calibrado | — | False | — |
| `--output-name` | 11_fusion_multivista | — | False | — |
| `--align-platform-axis` | True | — | False | Expresa la nube final en un marco canónico cuyo eje Y coincide con el eje físico de giro. Es una transformación rígida general. |
| `--prevoxel-mm` | 0.9 | — | False | — |
| `--fusion-voxel-mm` | 1.5 | — | False | — |
| `--minimum-core-support` | 2 | — | False | — |
| `--preserve-boundary-singletons` | True | — | False | — |
| `--singleton-neighbor-radius-cells` | 1 | — | False | — |
| `--singleton-minimum-core-neighbors` | 2 | — | False | — |
| `--singleton-minimum-confidence` | 0.42 | — | False | — |
| `--hierarchical-consensus` | True | — | False | — |
| `--preferred-anchor-support` | 5 | — | False | — |
| `--fallback-anchor-support` | 3 | — | False | — |
| `--minimum-anchor-points` | 1000 | — | False | — |
| `--minimum-anchor-ratio` | 0.03 | — | False | — |
| `--target-anchor-ratio` | 0.1 | — | False | Densidad mínima deseable del núcleo respecto a todos los voxels candidatos; permite bajar de soporte 5 a 4 sin asumir una forma. |
| `--minimum-anchor-extent-coverage` | 0.92 | — | False | — |
| `--anchor-continuity-radius-voxels` | 1.8 | — | False | — |
| `--anchor-minimum-local-neighbors` | 2 | — | False | — |
| `--minimum-anchor-local-continuity-ratio` | 0.55 | — | False | — |
| `--minimum-anchor-median-confidence` | 0.72 | — | False | — |
| `--minimum-anchor-median-normal-consistency` | 0.82 | — | False | — |
| `--maximum-anchor-spread-p90-mm` | 1.5 | — | False | — |
| `--minimum-extension-support` | 2 | — | False | — |
| `--extension-radius-one-level-voxels` | 1.15 | — | False | — |
| `--extension-radius-two-levels-voxels` | 0.85 | — | False | — |
| `--extension-radius-low-support-voxels` | 0.65 | — | False | — |
| `--extension-radius-voxels` | 0.0 | — | False | Compatibilidad: si es >0 reemplaza los tres radios adaptativos por un único radio uniforme. |
| `--minimum-incidence-cosine` | 0.15 | — | False | — |
| `--incidence-power` | 1.5 | — | False | — |
| `--depth-spread-reference-mm` | 6.0 | — | False | — |
| `--minimum-fused-confidence` | 0.32 | — | False | — |
| `--maximum-spread-for-full-score-mm` | 2.5 | — | False | — |
| `--local-surface-fusion` | True | — | False | Fusión robusta de parches entre voxels vecinos. |
| `--local-radius-voxels` | 2.5 | — | False | — |
| `--local-neighbors-per-pose` | 12 | — | False | — |
| `--local-normal-angle-deg` | 35.0 | — | False | — |
| `--local-sigma-floor-voxels` | 0.1 | — | False | — |
| `--local-max-shift-voxels` | 0.5 | — | False | — |
| `--minimum-independent-pose-separation` | 2 | — | False | Separación circular mínima (en índices de pose) para contar evidencia independiente. |
| `--minimum-independent-support-poses` | 2 | — | False | — |
| `--minimum-support-angular-span-poses` | 2 | — | False | — |
| `--minimum-unvalidated-support-poses` | 3 | — | False | — |
| `--minimum-unvalidated-confidence` | 0.68 | — | False | — |
| `--minimum-unvalidated-agreement` | 0.68 | — | False | — |
| `--maximum-conflict-pose-ratio` | 0.35 | — | False | — |
| `--minimum-heldout-pass-ratio` | 2.0 / 3.0 | — | False | Compatibilidad/diagnóstico. La decisión usa conteos enteros, no comparación flotante. |
| `--minimum-heldout-pass-numerator` | 2 | — | False | — |
| `--minimum-heldout-pass-denominator` | 3 | — | False | — |
| `--class2-local-coherence` | True | — | False | — |
| `--class2-coherence-small-neighbors` | 18 | — | False | — |
| `--class2-coherence-large-neighbors` | 36 | — | False | — |
| `--class2-coherence-min-neighbors` | 10 | — | False | — |
| `--class2-coherence-small-plane-p90-voxel` | 0.55 | — | False | — |
| `--class2-coherence-large-plane-p90-voxel` | 0.8 | — | False | — |
| `--class2-coherence-small-plane-uncertainty-factor` | 1.75 | — | False | — |
| `--class2-coherence-large-plane-uncertainty-factor` | 2.2 | — | False | — |
| `--class2-coherence-normal-p90-deg` | 42.0 | — | False | — |
| `--class2-coherence-scale-normal-deg` | 25.0 | — | False | — |
| `--class2-coherence-max-surface-variation` | 0.16 | — | False | — |
| `--class2-coherence-min-same-sheet-fraction` | 0.3 | — | False | — |
| `--class2-coherence-reject-flag-count` | 3 | — | False | Se rechaza clase 2 solo si acumula este número de fallos independientes. |
| `--class2-coherence-strict-max-flags` | 1 | — | False | Máximo de fallos para que una clase 2 pueda actuar como ancla fuerte. |
| `--class2-feature-min-family-fraction` | 0.18 | — | False | — |
| `--class2-feature-min-family-points` | 4 | — | False | — |
| `--class2-feature-max-within-family-p80-deg` | 17.0 | — | False | — |
| `--class2-feature-min-family-separation-deg` | 25.0 | — | False | — |
| `--class2-feature-min-tangent-backing-fraction` | 0.6 | — | False | — |
| `--observed-coverage-recovery` | True | — | False | — |
| `--coverage-recovery-iterations` | 6 | — | False | — |
| `--coverage-recovery-max-distance-voxels` | 3.0 | — | False | — |
| `--coverage-recovery-candidate-radius-voxels` | 2.75 | — | False | — |
| `--coverage-recovery-min-kept-neighbors` | 1 | — | False | — |
| `--coverage-recovery-min-candidate-neighbors` | 2 | — | False | — |
| `--coverage-recovery-normal-angle-deg` | 42.0 | — | False | — |
| `--coverage-recovery-tangent-residual-voxel` | 0.42 | — | False | — |
| `--coverage-recovery-tangent-uncertainty-factor` | 2.0 | — | False | — |
| `--coverage-recovery-min-support` | 2 | — | False | — |
| `--coverage-recovery-min-independent-support` | 2 | — | False | — |
| `--coverage-recovery-min-angular-span` | 2 | — | False | — |
| `--coverage-recovery-min-confidence` | 0.55 | — | False | — |
| `--coverage-recovery-min-agreement` | 0.55 | — | False | — |
| `--coverage-recovery-min-normal-consistency` | 0.82 | — | False | — |
| `--coverage-recovery-max-conflict-ratio` | 0.32 | — | False | — |
| `--coverage-recovery-max-uncertainty-voxel` | 0.7 | — | False | — |
| `--coverage-recovery-rejected-class2-min-score` | 0.58 | — | False | — |
| `--coverage-recovery-rejected-class2-max-red-flags` | 4 | — | False | — |
| `--coverage-recovery-rejected-class2-min-support` | 3 | — | False | — |
| `--coverage-recovery-rejected-class2-min-confidence` | 0.7 | — | False | — |
| `--coverage-recovery-max-fraction-of-initial` | 1.25 | — | False | — |
| `--coverage-recovery-min-coverage-improvement` | 0.005 | — | False | — |
| `--coverage-recovery-max-confidence-drop` | 0.06 | — | False | — |
| `--coverage-recovery-max-normal-consistency-drop` | 0.04 | — | False | — |
| `--coverage-recovery-max-uncertainty-p90-factor` | 1.35 | — | False | — |
| `--minimum-final-occupied-cell-retention` | 0.35 | — | False | V2: fracción mínima de celdas originales con TODOS sus candidatos a <=1 voxel de la selección final. El nombre de opción se conserva por compatibilidad; no es la razón entre cantidades de celdas finales y originales. |
| `--minimum-final-two-voxel-coverage` | 0.8 | — | False | Fracción mínima de candidatos observados que debe quedar a <=2 voxels de algún surfel retenido. Es cobertura de muestreo, no área física. |
| `--validated-pose-layer-separation` | True | — | False | — |
| `--validated-layer-min-supported-poses` | 4 | — | False | — |
| `--validated-layer-min-independent-per-layer` | 2 | — | False | — |
| `--validated-layer-min-separation-voxel` | 0.34 | — | False | — |
| `--validated-layer-uncertainty-factor` | 2.25 | — | False | — |
| `--validated-layer-within-sigma-factor` | 3.0 | — | False | — |
| `--validated-layer-min-winner-ratio` | 1.35 | — | False | — |
| `--validated-layer-min-independent-advantage` | 1 | — | False | — |
| `--validated-layer-max-shift-voxel` | 0.3 | — | False | — |
| `--validated-layer-max-shift-uncertainty-factor` | 1.5 | — | False | — |
| `--regional-pose-consensus` | True | — | False | — |
| `--regional-consensus-small-radius-voxels` | 3.2 | — | False | — |
| `--regional-consensus-large-radius-voxels` | 5.0 | — | False | — |
| `--regional-consensus-inner-radius-voxels` | 1.0 | — | False | — |
| `--regional-consensus-max-neighbors` | 112 | — | False | — |
| `--regional-consensus-min-neighbors` | 10 | — | False | — |
| `--regional-consensus-min-pose-neighbors` | 7 | — | False | — |
| `--regional-consensus-normal-angle-deg` | 28.0 | — | False | — |
| `--regional-consensus-max-normal-p90-deg` | 38.0 | — | False | — |
| `--regional-consensus-min-tangent-bins` | 5 | — | False | — |
| `--regional-consensus-fit-p90-voxel` | 0.6 | — | False | — |
| `--regional-consensus-fit-uncertainty-factor` | 2.8 | — | False | — |
| `--regional-consensus-scale-gate-voxel` | 0.2 | — | False | — |
| `--regional-consensus-scale-gate-uncertainty-factor` | 1.25 | — | False | — |
| `--regional-consensus-pose-dispersion-voxel` | 0.18 | — | False | — |
| `--regional-consensus-pose-dispersion-uncertainty-factor` | 1.2 | — | False | — |
| `--regional-consensus-min-independent-predictions` | 2 | — | False | — |
| `--regional-consensus-deadband-voxel` | 0.08 | — | False | — |
| `--regional-consensus-deadband-uncertainty-factor` | 0.45 | — | False | — |
| `--regional-consensus-max-shift-voxel` | 0.22 | — | False | — |
| `--regional-consensus-max-shift-uncertainty-factor` | 1.25 | — | False | — |
| `--regional-consensus-update-alpha` | 0.75 | — | False | — |
| `--regional-consensus-min-spacing-ratio` | 0.92 | — | False | — |
| `--regional-consensus-max-extent-change-fraction` | 0.0075 | — | False | — |
| `--raw-pose-bias-consensus` | True | — | False | — |
| `--raw-pose-bias-small-radius-voxels` | 4.0 | — | False | — |
| `--raw-pose-bias-large-radius-voxels` | 7.0 | — | False | — |
| `--raw-pose-bias-max-neighbors` | 180 | — | False | — |
| `--raw-pose-bias-min-patches-per-pose` | 5 | — | False | — |
| `--raw-pose-bias-min-independent-poses` | 3 | — | False | — |
| `--raw-pose-bias-normal-angle-deg` | 24.0 | — | False | — |
| `--raw-pose-bias-model-p90-voxel` | 0.22 | — | False | — |
| `--raw-pose-bias-model-p90-uncertainty-factor` | 1.6 | — | False | — |
| `--raw-pose-bias-scale-gate-voxel` | 0.14 | — | False | — |
| `--raw-pose-bias-scale-gate-uncertainty-factor` | 1.0 | — | False | — |
| `--raw-pose-bias-deadband-voxel` | 0.05 | — | False | — |
| `--raw-pose-bias-deadband-uncertainty-factor` | 0.35 | — | False | — |
| `--raw-pose-bias-max-shift-voxel` | 0.2 | — | False | — |
| `--raw-pose-bias-max-shift-uncertainty-factor` | 1.2 | — | False | — |
| `--raw-pose-bias-update-alpha` | 0.75 | — | False | — |
| `--raw-pose-bias-min-spacing-ratio` | 0.94 | — | False | — |
| `--raw-pose-bias-max-extent-change-fraction` | 0.005 | — | False | — |
| `--regional-fusion` | True | — | False | Conciliar parches solapados y verificar sus cambios contra cada pose. |
| `--geometry-refinement` | False | — | False | Diagnóstico/refinamiento opcional con modelos locales plano/cilindro/cuadrático. Desactivado por defecto: la salida normal permanece estrictamente agnóstica a primitivas. |
| `--independent-patches` | True | — | False | — |
| `--complete-gaps` | True | — | False | Genera solo guías locales que superen validación multivista; no son mediciones. |
| `--complete-bottom` | False | — | False | Hipótesis opcional de cierre terminal. Desactivada por defecto por ser inferida. |
| `--completion-max-span-mm` | 0.0 | — | False | 0: límite automático estrictamente local = max(6 vox, min(12 vox, 18%% de la extensión robusta)). Se aplica también a cierres terminales explícitos. |
| `--completion-max-points` | 12000 | — | False | — |
| `--completion-silhouette-tolerance-px` | 2 | — | False | — |
| `--completion-min-silhouette-tested-poses` | 8 | — | False | — |
| `--completion-min-silhouette-inside-ratio` | 0.9 | — | False | — |
| `--completion-max-silhouette-contradictions` | 1 | — | False | — |
| `--completion-depth-pixel-radius` | 3.0 | — | False | — |
| `--completion-depth-neighbors` | 8 | — | False | — |
| `--completion-depth-noise-floor-mm` | 1.0 | — | False | — |
| `--completion-depth-uncertainty-factor` | 2.0 | — | False | — |
| `--completion-min-depth-tested-poses` | 2 | — | False | — |
| `--completion-min-independent-depth-agreements` | 2 | — | False | — |
| `--completion-max-free-space-contradictions` | 0 | — | False | — |
| `--completion-min-independent-pose-separation` | 2 | — | False | — |
| `--completion-min-retained-fraction` | 0.7 | — | False | — |
| `--completion-max-observation-distance-voxels` | 3.0 | — | False | — |

## [12_regularizar_nube.py](../procesamiento/12_regularizar_nube.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | sin valor explícito | — | True | — |
| `--source` | 11_fusion_multivista | — | False | — |
| `--output-name` | 12_regularizacion_nube | — | False | — |
| `--sor-neighbors` | 24 | — | False | — |
| `--sor-std-ratio` | 2.5 | — | False | — |
| `--radius-neighbors` | 5 | — | False | — |
| `--radius-mm` | 0.0 | — | False | 0=auto: max(3.5, 2.5*voxel) |
| `--always-preserve-support` | 3 | — | False | — |
| `--always-preserve-confidence` | 0.72 | — | False | — |
| `--minimum-input-evidence-class` | 2 | — | False | — |
| `--minimum-input-independent-support` | 2 | — | False | — |
| `--minimum-input-angular-span-poses` | 2 | — | False | — |
| `--maximum-input-conflict-pose-ratio` | 0.35 | — | False | — |
| `--minimum-input-heldout-pass-ratio` | 2.0 / 3.0 | — | False | Compatibilidad/diagnóstico. La decisión de producción usa conteos enteros para evitar que 2/3 falle frente a 0.67. |
| `--minimum-input-heldout-pass-numerator` | 2 | — | False | — |
| `--minimum-input-heldout-pass-denominator` | 3 | — | False | — |
| `--heldout-rescue-min-independent-support` | 3 | — | False | Respaldo independiente mínimo para rescatar un voto held-out insuficiente. |
| `--heldout-rescue-min-angular-span-poses` | 3 | — | False | — |
| `--heldout-rescue-max-conflict-pose-ratio` | 0.2 | — | False | — |
| `--heldout-rescue-error-uncertainty-factor` | 1.0 | — | False | Error mediano held-out máximo relativo a la incertidumbre local. |
| `--heldout-rescue-p90-uncertainty-factor` | 1.5 | — | False | P90 held-out máximo relativo a la incertidumbre local. |
| `--heldout-uncertainty-floor-voxel-factor` | 0.12 | — | False | Piso de incertidumbre para normalizar los errores held-out. |
| `--pose-shared-support-strong` | 2 | — | False | — |
| `--pose-shared-support-weak-weight` | 0.3 | — | False | — |
| `--pose-no-shared-support-weight` | 0.05 | — | False | — |
| `--geometric-cleanup` | True | — | False | — |
| `--adjacency-radius-mm` | 0.0 | — | False | 0=auto según voxel y spacing de la nube preseleccionada. |
| `--adjacency-voxel-factor` | 2.25 | — | False | — |
| `--adjacency-spacing-factor` | 5.5 | — | False | — |
| `--component-dbscan-min-points` | 4 | — | False | — |
| `--density-radius-mm` | 0.0 | — | False | 0=auto; radio para medir respaldo espacial local. |
| `--density-radius-voxel-factor` | 2.8 | — | False | — |
| `--density-radius-spacing-factor` | 7.0 | — | False | — |
| `--density-min-neighbors` | 6 | — | False | — |
| `--density-high-quality-min-neighbors` | 3 | — | False | — |
| `--component-large-fraction` | 0.01 | — | False | Fracción que hace significativo un componente por tamaño. |
| `--component-min-points` | 48 | — | False | — |
| `--component-score-threshold` | 0.62 | — | False | — |
| `--component-far-score-penalty` | 0.14 | — | False | — |
| `--component-high-evidence-score` | 0.82 | — | False | — |
| `--component-near-distance-mm` | 0.0 | — | False | 0=auto según radio de adyacencia y escala robusta. |
| `--component-near-adjacency-factor` | 6.0 | — | False | — |
| `--component-near-diagonal-ratio` | 0.06 | — | False | — |
| `--maximum-component-diagnostics` | 250 | — | False | — |
| `--surface-regularization` | True | — | False | — |
| `--smooth-iterations` | 2 | — | False | — |
| `--smooth-knn` | 36 | — | False | — |
| `--smooth-radius-mm` | 0.0 | — | False | 0=auto según spacing y voxel. |
| `--smooth-radius-spacing-factor` | 3.2 | — | False | — |
| `--smooth-spatial-sigma-factor` | 0.52 | — | False | — |
| `--smooth-normal-sigma-deg` | 16.0 | — | False | — |
| `--feature-protect-angle-deg` | 21.0 | — | False | Dispersión local de normales a partir de la cual se reduce progresivamente el suavizado. |
| `--feature-min-strength` | 0.1 | — | False | — |
| `--boundary-asymmetry-low` | 0.12 | — | False | — |
| `--boundary-asymmetry-high` | 0.34 | — | False | — |
| `--boundary-min-strength` | 0.12 | — | False | — |
| `--smooth-strength` | 0.64 | — | False | — |
| `--max-shift-per-iteration-spacing` | 0.42 | — | False | — |
| `--max-total-shift-spacing` | 1.05 | — | False | — |
| `--minimum-smoothing-neighbors` | 10 | — | False | — |
| `--height-robust-scale-spacing-floor` | 0.1 | — | False | — |
| `--height-robust-sigma-factor` | 2.6 | — | False | — |
| `--minimum-spacing-ratio-after-smoothing` | 0.82 | — | False | El spacing NN mediano tras suavizado no puede caer por debajo de esta fracción del spacing filtrado previo. Si cae, se reduce adaptativamente la intensidad efectiva de regularización. |
| `--spacing-guard-bisection-steps` | 10 | — | False | — |
| `--spacing-guard-sample-points` | 30000 | — | False | — |
| `--support-neighbor-weight-exponent` | 0.35 | — | False | — |
| `--confidence-neighbor-weight-exponent` | 0.65 | — | False | — |
| `--quality-anchor-strength` | 0.28 | — | False | Puntos de soporte/confianza alta se mueven ligeramente menos; la procedencia por pose evita mezclar capas incompatibles; no quedan congelados. |
| `--normal-pca-knn` | 30 | — | False | — |
| `--normal-pca-radius-factor` | 3.4 | — | False | — |
| `--normal-radius-mm` | 0.0 | — | False | 0=auto para orientación final Open3D. |
| `--normal-max-nn` | 50 | — | False | — |
| `--normal-orientation-k` | 30 | — | False | — |
| `--roughness-evaluation-knn` | 28 | — | False | — |
| `--maximum-diagnostic-points` | 60000 | — | False | — |
| `--warning-extent-change-ratio` | 0.025 | — | False | Warning si un extent robusto cambia >2.5%%. |

## [13_reconstruir_superficie.py](../procesamiento/13_reconstruir_superficie.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | sin valor explícito | — | True | — |
| `--source` | 12_regularizacion_nube | — | False | — |
| `--output-name` | 13_reconstruccion_superficie | — | False | — |
| `--permissive-completion` | None | — | False | Compatibilidad: fuerza o desactiva el relleno local. |
| `--completion-mode` | auto | ('auto', 'always', 'never') | False | — |
| `--completion-min-contours` | 20 | — | False | argparse.SUPPRESS |
| `--completion-min-boundary-ratio` | 0.04 | — | False | argparse.SUPPRESS |
| `--completion-min-perimeter-ratio` | 2.0 | — | False | argparse.SUPPRESS |
| `--local-hole-max-diameter-mm` | 0.0 | — | False | 0 evalua aberturas de cualquier tamano; un valor positivo limita su diametro |
| `--local-hole-max-distance-mm` | 2.0 | — | False | — |
| `--complete-walls-open-base` | True | — | False | — |
| `--wall-voxel-mm` | 0.7 | — | False | — |
| `--threads` | 0 | — | False | 0 usa todos los procesadores lógicos disponibles. |
| `--poisson-depth` | 0 | — | False | 0 calcula depth automáticamente. |
| `--poisson-min-depth` | 8 | — | False | — |
| `--poisson-max-depth` | 8 | — | False | — |
| `--poisson-scale` | 1.08 | — | False | — |
| `--poisson-target-cell-spacing-factor` | 0.9 | — | False | — |
| `--poisson-min-cell-mm` | 0.45 | — | False | — |
| `--poisson-density-trim-quantile` | 0.01 | — | False | Recorte conservador de densidad Poisson. 0 lo desactiva; el valor por defecto elimina únicamente el 1%% de menor densidad antes de aplicar los gates de evidencia observacional. |
| `--bbox-margin-mm` | 4.0 | — | False | — |
| `--boundary-retraction` | True | — | False | Replegar fronteras extrapoladas con evidencia local y guardas geométricas. |
| `--max-poisson-input-points` | 55000 | — | False | — |
| `--input-voxel-spacing-factor` | 1.1 | — | False | — |
| `--meshing-minimum-voxel-mm` | 1.0 | — | False | — |
| `--meshing-confidence-exponent` | 1.5 | — | False | — |
| `--meshing-support-exponent` | 0.65 | — | False | — |
| `--require-evidence-contract` | True | — | False | — |
| `--minimum-evidence-class` | 2 | — | False | — |
| `--minimum-independent-support` | 2 | — | False | — |
| `--minimum-angular-span-poses` | 2 | — | False | — |
| `--maximum-conflict-pose-ratio` | 0.35 | — | False | — |
| `--minimum-heldout-pass-ratio` | 0.67 | — | False | — |
| `--evidence-strength-exponent` | 1.0 | — | False | — |
| `--component-min-triangles` | 60 | — | False | — |
| `--component-min-triangle-ratio` | 0.00025 | — | False | — |
| `--support-trim` | True | — | False | Conserva el nombre por compatibilidad. Solo permite eliminar componentes completas aisladas; nunca caras internas. |
| `--support-strong-distance-mm` | 0.0 | — | False | 0 calcula automáticamente la distancia de respaldo fuerte. |
| `--support-reject-distance-mm` | 0.0 | — | False | 0 calcula la distancia donde la regularización llega al máximo. |
| `--support-min-confidence` | 0.55 | — | False | — |
| `--support-min-views` | 2.0 | — | False | — |
| `--support-component-min-seed-faces` | 12 | — | False | — |
| `--support-component-min-seed-ratio` | 0.02 | — | False | — |
| `--support-max-removed-triangle-ratio` | 0.45 | — | False | argparse.SUPPRESS |
| `--weak-region-smoothing` | True | — | False | — |
| `--weak-region-iterations` | 5 | — | False | — |
| `--weak-region-lambda` | 0.34 | — | False | — |
| `--weak-region-mu` | -0.35 | — | False | — |
| `--weak-region-anchor-confidence` | 0.72 | — | False | — |
| `--weak-region-anchor-support` | 3.0 | — | False | — |
| `--weak-region-reliability-weight` | 0.35 | — | False | — |
| `--weak-region-max-displacement-mm` | 0.0 | — | False | 0 calcula un límite físico a partir del spacing de Poisson. |
| `--bpa-fallback` | True | — | False | BPA se usa como rescate y también puede competir cuando Poisson muestra extrapolación no respaldada. |
| `--bpa-radius-factors` | 1.5,2.5,4.0 | — | False | — |
| `--adaptive-bpa-competition` | True | — | False | Evalúa BPA además de Poisson solo cuando la malla implícita muestra riesgo de extrapolación. La selección usa evidencia geométrica, no el nombre ni la forma conocida del objeto. |
| `--candidate-min-absolute-coverage` | 0.9 | — | False | — |
| `--candidate-min-relative-coverage` | 0.94 | — | False | — |
| `--poisson-risk-p90-spacing-factor` | 4.0 | — | False | — |
| `--poisson-risk-p95-spacing-factor` | 5.0 | — | False | — |
| `--poisson-risk-bbox-expansion-ratio` | 0.05 | — | False | — |
| `--candidate-switch-min-score-gain` | 0.08 | — | False | — |
| `--candidate-min-largest-component-ratio` | 0.25 | — | False | Guardia de seguridad para alternativas fragmentadas. No exige una sola componente dominante: la fragmentación se penaliza en la puntuación y solo se rechaza si es extrema. |
| `--candidate-max-components` | 20 | — | False | — |
| `--candidate-max-boundary-edge-ratio` | 0.15 | — | False | — |
| `--adaptive-observed-refinement` | True | — | False | Si ningún candidato cumple, repetir BPA con menos reducción de muestras observadas. |
| `--evaluation-cloud-samples` | 30000 | — | False | — |
| `--evaluation-mesh-samples` | 30000 | — | False | — |
| `--coverage-gate-mm` | 3.0 | — | False | — |
| `--warning-max-components` | 80 | — | False | — |
| `--warning-min-largest-component-ratio` | 0.94 | — | False | — |
| `--warning-max-mesh-to-cloud-p90-mm` | 3.5 | — | False | — |
| `--use-estimated-completion` | False | — | False | — |
| `--completion-source` | 11_fusion_multivista | — | False | — |

## [14_limpiar_topologia.py](../procesamiento/14_limpiar_topologia.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | sin valor explícito | — | True | — |
| `--mesh-source` | 13_reconstruccion_superficie | — | False | — |
| `--cloud-source` | 12_regularizacion_nube | — | False | — |
| `--output-name` | 14_limpieza_topologica | — | False | — |
| `--require-evidence-contract` | True | — | False | — |
| `--minimum-evidence-class` | 2 | — | False | — |
| `--minimum-independent-support` | 2 | — | False | — |
| `--minimum-angular-span-poses` | 2 | — | False | — |
| `--maximum-conflict-pose-ratio` | 0.35 | — | False | — |
| `--minimum-heldout-pass-ratio` | 0.67 | — | False | — |
| `--tiny-component-max-triangles` | 20 | — | False | — |
| `--tiny-component-max-area-ratio` | 0.0005 | — | False | — |
| `--tiny-component-max-extent-spacing-factor` | 8.0 | — | False | — |
| `--tiny-component-min-median-support-to-keep` | 1.5 | — | False | — |
| `--tiny-component-min-median-confidence-to-keep` | 0.68 | — | False | — |
| `--post-repair-component-cleanup` | True | — | False | — |
| `--post-repair-source-ancestry-min-ratio` | 0.6 | — | False | — |
| `--post-repair-dominant-area-ratio` | 20.0 | — | False | La componente hermana dominante debe tener al menos esta relación de área para considerar la menor un fragmento desprendido. |
| `--orientation-support-cap` | 5.0 | — | False | — |
| `--orientation-support-weight` | 0.9 | — | False | — |
| `--orientation-confidence-weight` | 1.35 | — | False | — |
| `--orientation-area-weight` | 0.25 | — | False | — |
| `--orientation-interior-weight` | 0.25 | — | False | — |
| `--max-orientation-face-removal-ratio` | 0.02 | — | False | Máximo 2%% de caras de la malla tras limpieza de componentes. |
| `--reject-if-self-intersecting` | True | — | False | Diagnóstico raw. V2.0 usa clasificación semántica para el gate. |
| `--semantic-geometric-epsilon-spacing-factor` | 0.0001 | — | False | — |
| `--semantic-contact-locality-spacing-factor` | 0.001 | — | False | — |
| `--semantic-repair-max-iterations` | 4 | — | False | — |
| `--fill-existing-holes` | False | — | False | Relleno inferido opcional de huecos preexistentes; no es evidencia observada. |
| `--existing-hole-max-diameter-mm` | 12.0 | — | False | — |
| `--repair-holes` | True | — | False | — |
| `--estimated-terminal-caps` | False | — | False | Compatibilidad con lanzadores anteriores: las tapas estimadas permanecen retiradas en V2.0. |
| `--repair-hole-max-diameter-spacing` | 8.0 | — | False | — |
| `--closure-planarity-ratio` | 0.04 | — | False | — |
| `--closure-max-loop-vertices` | 512 | — | False | — |
| `--closure-max-new-triangles` | 30000 | — | False | — |
| `--refine-terminal-caps` | True | — | False | — |
| `--rim-max-shift-spacing` | 1.0 | — | False | — |

## [15_pulir_modelo.py](../procesamiento/15_pulir_modelo.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | sin valor explícito | — | True | — |
| `--mesh-source` | 14_limpieza_topologica | — | False | — |
| `--cloud-source` | 12_regularizacion_nube | — | False | — |
| `--topology-source` | 14_limpieza_topologica | — | False | Fuente del resumen Paso 14. Su estimated_point_spacing_mm se usa para coincident-lock y para la guarda semántica, con la misma escala física que usa Paso 14. |
| `--output-name` | 15_pulido_final | — | False | — |
| `--require-evidence-contract` | True | — | False | — |
| `--minimum-evidence-class` | 2 | — | False | — |
| `--minimum-independent-support` | 2 | — | False | — |
| `--minimum-angular-span-poses` | 2 | — | False | — |
| `--maximum-conflict-pose-ratio` | 0.35 | — | False | — |
| `--minimum-heldout-pass-ratio` | 0.67 | — | False | — |
| `--normal-trend-enabled` | True | — | False | — |
| `--normal-filter-iterations` | 5 | — | False | — |
| `--normal-filter-angle-sigma-deg` | 20.0 | — | False | — |
| `--normal-filter-max-cross-angle-deg` | 60.0 | — | False | — |
| `--normal-filter-self-weight` | 1.2 | — | False | — |
| `--normal-filter-edge-weight-floor` | 0.32 | — | False | Evita que una saliente de escala media quede totalmente congelada por el detector feature-aware. Las discontinuidades reales siguen bloqueadas por la diferencia angular bilateral. |
| `--normal-trend-cycles` | 1 | — | False | Un ciclo suele ser suficiente; las guardas posteriores limitan cualquier corrección excesiva. |
| `--normal-trend-small-radius-spacing` | 4.0 | — | False | — |
| `--normal-trend-large-radius-spacing` | 14.0 | — | False | Escala grande usada como tendencia. Sigue siendo local y se expresa en múltiplos del spacing físico. |
| `--normal-trend-max-neighbors` | 500 | — | False | — |
| `--normal-trend-query-chunk` | 1200 | — | False | Número de vértices consultados simultáneamente en KDTree. Mantiene bajo el uso de RAM aun con vecindarios grandes. |
| `--normal-trend-min-small-neighbors` | 16 | — | False | — |
| `--normal-trend-min-large-neighbors` | 32 | — | False | — |
| `--normal-trend-neighbor-angle-deg` | 34.0 | — | False | Vecinos con normales muy distintas se excluyen del ajuste, evitando cruzar esquinas/aristas reales. |
| `--normal-trend-irls-iterations` | 3 | — | False | — |
| `--normal-trend-huber-k` | 1.5 | — | False | — |
| `--normal-trend-scale-difference-threshold-spacing` | 0.035 | — | False | Diferencia mínima entre predicción pequeña/grande antes de considerar que existe detalle de escala media. |
| `--normal-trend-scale-difference-transition-spacing` | 0.16 | — | False | — |
| `--normal-trend-strength` | 0.88 | — | False | — |
| `--normal-trend-max-shift-per-cycle-spacing` | 0.48 | — | False | — |
| `--normal-trend-max-total-shift-spacing` | 0.75 | — | False | — |
| `--despike-enabled` | True | — | False | — |
| `--despike-rings` | 3 | — | False | Número de anillos topológicos usados para la superficie local. |
| `--despike-min-neighbors` | 18 | — | False | — |
| `--despike-zscore-threshold` | 3.0 | — | False | Umbral robusto \|residuo\|/(1.4826*MAD). |
| `--despike-isolation-ratio` | 0.58 | — | False | Fracción máxima de vecinos que también pueden ser outliers del mismo signo para considerar la saliente como aislada. |
| `--despike-correction-strength` | 0.84 | — | False | — |
| `--despike-max-shift-spacing` | 0.9 | — | False | — |
| `--despike-feature-strength-min` | 0.38 | — | False | Solo se corrigen vértices cuya protección feature-aware sea al menos este valor. |
| `--despike-irls-iterations` | 3 | — | False | — |
| `--despike-huber-k` | 1.5 | — | False | — |
| `--iterations` | 6 | — | False | — |
| `--lambda-factor` | 0.44 | — | False | — |
| `--mu-factor` | -0.29 | — | False | — |
| `--normal-component` | 1.0 | — | False | — |
| `--tangential-component` | 0.1 | — | False | — |
| `--surface-fairing-enabled` | True | — | False | Activa el pulido residual por parches cuadráticos locales. No supone ninguna forma geométrica. |
| `--surface-fairing-cycles` | 4 | — | False | — |
| `--surface-fairing-rings` | 5 | — | False | Radio topológico del parche. Cinco anillos eliminan grumos de varias celdas sin convertir el ajuste en una forma global. |
| `--surface-fairing-min-neighbors` | 24 | — | False | — |
| `--surface-fairing-neighbor-angle-deg` | 32.0 | — | False | Impide que el ajuste atraviese aristas o pliegues reales. |
| `--surface-fairing-irls-iterations` | 3 | — | False | — |
| `--surface-fairing-huber-k` | 1.35 | — | False | — |
| `--surface-fairing-activation-threshold-spacing` | 0.018 | — | False | Residuo normal mínimo antes de intervenir, relativo al spacing. |
| `--surface-fairing-activation-transition-spacing` | 0.1 | — | False | — |
| `--surface-fairing-noise-scale-factor` | 0.45 | — | False | Evita perseguir variaciones menores que el ruido robusto local. |
| `--surface-fairing-strength` | 0.72 | — | False | — |
| `--surface-fairing-feature-strength-min` | 0.12 | — | False | Por debajo de esta movilidad la arista se considera persistente y queda inmóvil. |
| `--surface-fairing-max-shift-per-cycle-spacing` | 0.24 | — | False | — |
| `--surface-fairing-max-total-shift-spacing` | 1.05 | — | False | — |
| `--surface-fairing-convergence-spacing` | 0.004 | — | False | Detiene ciclos extra cuando el P90 del movimiento es menor. |
| `--feature-soft-angle-deg` | 28.0 | — | False | — |
| `--feature-hard-angle-deg` | 65.0 | — | False | — |
| `--feature-min-strength` | 0.05 | — | False | — |
| `--boundary-ring-strength` | 0.3 | — | False | — |
| `--edge-crossing-min-weight` | 0.04 | — | False | — |
| `--feature-guide-iterations` | 4 | — | False | Suavizado SOLO de una copia guía usada para distinguir rugosidad de alta frecuencia de aristas persistentes. |
| `--feature-guide-lambda` | 0.38 | — | False | — |
| `--max-shift-per-pass-spacing` | 0.28 | — | False | — |
| `--max-total-shift-spacing` | 0.85 | — | False | — |
| `--max-robust-extent-change-ratio` | 0.012 | — | False | Máximo cambio permitido en extent P01-P99 por eje (1.2%%). |
| `--max-orientation-flip-ratio` | 0.0 | — | False | V1.1: no se permite ningún volteo de triángulo. |
| `--safety-bisection-steps` | 14 | — | False | — |
| `--safety-local-rollback-rings` | 2 | — | False | — |
| `--safety-local-max-expansions` | 8 | — | False | — |
| `--safety-local-ring-floor` | 1.0 | — | False | V1.8: a beta=0 toda la banda afectada vuelve exactamente a la geometría segura; evita fallback global alpha=0. |
| `--semantic-intersection-guard` | True | — | False | — |
| `--semantic-guard-bisection-steps` | 16 | — | False | Bisección del alpha LOCAL. También se usa en el fallback global de seguridad. |
| `--semantic-local-rollback-rings` | 2 | — | False | Anillos de transición alrededor de los triángulos que participan en una intersección. |
| `--semantic-local-rollback-max-expansions` | 8 | — | False | Máximo de expansiones de la región si el rollback local desplaza el conflicto hacia su frontera. |
| `--semantic-local-rollback-ring-floor` | 1.0 | — | False | Peso mínimo de rollback en el anillo exterior. Los vértices directamente conflictivos siempre tienen peso 1. |
| `--semantic-geometric-epsilon-spacing-factor` | 0.0001 | — | False | — |
| `--semantic-contact-locality-spacing-factor` | 0.001 | — | False | — |
| `--residual-finish` | True | — | False | Acabado residual local tras el pulido optimizado. |
| `--residual-finish-cycles` | 2 | (1, 2) | False | — |
| `--residual-finish-strength` | 0.6 | — | False | — |
| `--residual-finish-max-shift-spacing` | 0.28 | — | False | Máximo desplazamiento adicional como fracción del spacing; hasta 0.35. |
| `--cloud-evaluation-samples` | 40000 | — | False | — |
| `--preview-max-vertices` | 70000 | — | False | — |

## [16_validar_intersecciones.py](../procesamiento/16_validar_intersecciones.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | sin valor explícito | — | True | — |
| `--mesh-source` | 15_pulido_final | — | False | — |
| `--topology-source` | 14_limpieza_topologica | — | False | Fuente del resumen topológico V1.3. La geometría a validar puede venir del paso 15 sin perder la escala del paso 14. |
| `--output-name` | 16_validacion_intersecciones | — | False | — |
| `--geometric-epsilon-spacing-factor` | 0.0001 | — | False | Tolerancia para posiciones coincidentes. Default = 0.0001 x spacing. |
| `--contact-locality-spacing-factor` | 0.001 | — | False | Tolerancia para confirmar que la intersección completa está limitada al contacto coincidente. |

## [17_validar_modelo.py](../procesamiento/17_validar_modelo.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | sin valor explícito | — | True | — |
| `--cloud-source` | 12_regularizacion_nube | — | False | — |
| `--mesh-source` | 15_pulido_final | — | False | — |
| `--intersection-source` | 16_validacion_intersecciones | — | False | — |
| `--output-name` | 17_validacion_modelo | — | False | — |
| `--coverage-gate-mm` | 3.0 | — | False | — |
| `--warning-coverage` | 0.95 | — | False | — |
| `--reject-coverage` | 0.85 | — | False | — |
| `--warning-cloud-mesh-p90-mm` | 2.5 | — | False | — |
| `--reject-cloud-mesh-p90-mm` | 4.5 | — | False | — |
| `--warning-mesh-cloud-p90-mm` | 2.0 | — | False | — |
| `--reject-mesh-cloud-p90-mm` | 4.0 | — | False | — |
| `--adaptive-mesh-cloud-thresholds` | True | — | False | — |
| `--warning-mesh-cloud-spacing-factor` | 2.5 | — | False | — |
| `--reject-mesh-cloud-spacing-factor` | 5.0 | — | False | — |
| `--maximum-adaptive-reject-mesh-cloud-p90-mm` | 6.0 | — | False | Techo absoluto del umbral adaptativo mesh->cloud. El valor nunca puede quedar por debajo de --reject-mesh-cloud-p90-mm. |
| `--allow-coherent-interpolation` | False | — | False | Compatibilidad diagnóstica. La coherencia puede documentarse, pero nunca anula el umbral duro mesh->cloud de rechazo. |
| `--coherent-interpolation-min-coverage` | 0.95 | — | False | — |
| `--coherent-interpolation-max-bbox-expansion-ratio` | 1.12 | — | False | — |
| `--coherent-interpolation-min-largest-area-ratio` | 0.97 | — | False | — |
| `--coherent-interpolation-max-boundary-edge-ratio` | 0.05 | — | False | — |
| `--coherent-interpolation-max-p95-extent-ratio` | 0.12 | — | False | Máximo P95 mesh->cloud dividido por la mayor extensión robusta de la nube. Es independiente del tipo y tamaño del objeto. |
| `--warning-bbox-expansion-ratio` | 1.08 | — | False | — |
| `--reject-bbox-expansion-ratio` | 1.18 | — | False | — |
| `--warning-largest-component-area-ratio` | 0.97 | — | False | — |
| `--reject-largest-component-area-ratio` | 0.85 | — | False | — |
| `--warning-boundary-edge-ratio` | 0.03 | — | False | — |
| `--reject-boundary-edge-ratio` | 0.15 | — | False | — |
| `--cloud-samples` | 100000 | — | False | — |
| `--mesh-samples` | 100000 | — | False | — |
| `--require-evidence-contract` | True | — | False | — |
| `--strong-core-support` | 4 | — | False | Compatibilidad legacy; no gobierna el núcleo V1.8 cuando existe contrato de evidencia. |
| `--strong-core-min-evidence-class` | 3 | — | False | — |
| `--strong-core-min-independent-support` | 2 | — | False | — |
| `--strong-core-min-angular-span-poses` | 2 | — | False | — |
| `--strong-core-max-conflict-pose-ratio` | 0.2 | — | False | — |
| `--strong-core-min-heldout-pass-ratio` | 0.8 | — | False | — |
| `--strong-core-minimum-points` | 500 | — | False | — |
| `--warning-observation-envelope-expansion-ratio` | 1.1 | — | False | — |
| `--reject-observation-envelope-expansion-ratio` | 1.22 | — | False | — |

## [18_exportar_modelo_blender.py](../procesamiento/18_exportar_modelo_blender.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--root` | sin valor explícito | — | True | — |
| `--object` | sin valor explícito | — | True | — |
| `--mesh-source` | 15_pulido_final | — | False | — |
| `--validation-source` | 17_validacion_modelo | — | False | — |
| `--pre-polish-source` | 14_limpieza_topologica | — | False | Fuente topológica anterior al pulido final. Se conserva para cuantificar el efecto exclusivo del paso 15 V1.8. |
| `--pre-polish-fallback-source` | 14_limpieza_topologica | — | False | Fallback canónico al mismo producto topológico del paso 14 V1.3. |
| `--output-name` | 18_exportacion_modelo | — | False | — |
| `--final-folder-name` | resultado_final | — | False | — |
| `--blender-scale` | 0.001 | — | False | Conversión de mm científicos a unidades Blender. 0.001 => metros. |
| `--require-evidence-aware-validation` | True | — | False | Exige que paso 17 haya validado el núcleo multivista usando evidencia independiente/held-out, no solo conteo bruto de poses. |

## [auditar_evidencia_lr.py](../herramientas/auditar_evidencia_lr.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--depth-dir` | sin valor explícito | — | True | — |
| `--regional-dir` |  | — | False | — |
| `--stem` | sin valor explícito | — | True | — |
| `--object-mask` |  | — | False | — |

## [calibrar_estereo_checkerboard.py](../herramientas/calibrar_estereo_checkerboard.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--left-camera` | DEFAULT_LEFT_CAMERA | — | False | — |
| `--right-camera` | DEFAULT_RIGHT_CAMERA | — | False | — |
| `--width` | DEFAULT_WIDTH | — | False | — |
| `--height` | DEFAULT_HEIGHT | — | False | — |
| `--cols` | DEFAULT_COLS | — | False | Esquinas internas horizontales. |
| `--rows` | DEFAULT_ROWS | — | False | Esquinas internas verticales. |
| `--square-mm` | DEFAULT_SQUARE_MM | — | False | Lado físico real de cada cuadro. |
| `--dataset` | Path('calibracion_checkerboard') | — | False | — |
| `--output` | Path('calibracion_estereo_candidata') | — | False | — |
| `--calibrate-only` | sin valor explícito | — | False | Usar imágenes ya capturadas. |
| `--expected-baseline-mm` | DEFAULT_EXPECTED_BASELINE_MM | — | False | Baseline físico aproximado del montaje. Usa 0 para desactivar la comprobación. |
| `--baseline-tolerance-mm` | DEFAULT_BASELINE_TOLERANCE_MM | — | False | — |

## [comparar_dimensiones.py](../herramientas/comparar_dimensiones.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--results` | ROOT / 'resultados' | — | False | — |

## [preparar_resultados_publicos.py](../herramientas/preparar_resultados_publicos.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--output` | ROOT / 'resultados' | — | False | — |

## [promover_calibracion_plataforma.py](../herramientas/promover_calibracion_plataforma.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `mode` | sin valor explícito | ['evaluar', 'activar'] | False | — |
| `--candidate-dir` | sin valor explícito | — | True | resultado_calibracion_plataforma de la campaña de calibración |
| `--evidence` | sin valor explícito | — | False | Informe independiente opcional; sin él, activar utiliza los controles del paso 09 |

## [registrar_entorno.py](../herramientas/registrar_entorno.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--output` | sin valor explícito | — | False | JSON de destino; sin esta opción imprime el registro. |

## [reparar_manifiestos_exportacion.py](../herramientas/reparar_manifiestos_exportacion.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--jobs` | ROOT / 'trabajos' | — | False | — |

## [verificar_dependencias.py](../herramientas/verificar_dependencias.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--quiet` | sin valor explícito | — | False | Reduce la salida en consola. |
| `--json` |  | — | False | Guarda el reporte de verificación en JSON. |
| `--no-install` | sin valor explícito | — | False | Solo verifica; no instala paquetes faltantes. |
| `--startup` | sin valor explícito | — | False | Permite abrir la interfaz para reparar una calibración ausente o inválida. |

## [verificar_informe_calibracion.py](../herramientas/verificar_informe_calibracion.py)

| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |
| --- | --- | --- | --- | --- |
| `--candidate` | sin valor explícito | — | True | — |
| `--evidence` | sin valor explícito | — | True | — |
