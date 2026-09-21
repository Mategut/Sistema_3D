# Revisión integral del pipeline 3D — 18/09/2026

## Objetivo
Corregir las causas que permitían generar una malla continua pero geométricamente poco respaldada por las observaciones. Las modificaciones son generales y no introducen reglas específicas para cubos, cilindros, pirámides ni otras geometrías.

## Cambios aplicados

### Paso 02 — profundidad CREStereo
- La auditoría epipolar SIFT queda diagnóstica por defecto.
- `--epipolar-auto-correct` ahora está desactivado por defecto.
- Una tendencia SIFT ya no deforma silenciosamente la imagen derecha rectificada ni deja P1/P2/Q incoherentes.
- Se añadió `--epipolar-audit-strict` para detener el paso si se desea convertir la auditoría en gate duro.
- La entrada 1920x1080 ya no se deforma directamente a 640x480. Se usa letterbox con escala isotrópica y se revierte correctamente la disparidad al tamaño original.

### Paso 04 — validación de disparidad
- Se restauró la validación regional robusta anterior a la simplificación que aceptaba prácticamente cualquier disparidad finita dentro del rango físico.
- Se recuperaron confianza, SLIC, RANSAC regional, consistencia local, consenso de observaciones y clasificación explícita de fuentes.
- El objetivo vuelve a ser conservar observaciones reales y rechazar outliers con evidencia conjunta, sin completar una forma conocida.

### Paso 08 — registro de referencia para calibrar plataforma
- Eliminada la baseline fija de 81.0558 mm.
- La baseline se deriva directamente de `stereo_initial.yaml` y se verifica contra P2 y Q.
- El pipeline pasa explícitamente la calibración estéreo vigente al Paso 08.

### Paso 09 — congelación de calibración de plataforma
- La baseline se toma exclusivamente de la calibración estéreo vigente.
- Se rechaza un registro de referencia si fue calculado con otra baseline.
- Se guarda un `stereo_calibration_contract` con SHA-256 de `stereo_initial.yaml` y `rectification_maps.npz`, baseline, focal rectificada y resolución geométrica.
- Se evita asociar hashes de una calibración nueva con geometría calculada usando una baseline antigua.

### Calibración de plataforma incluida en el ZIP
- La calibración antigua queda marcada como `invalidated_stereo_geometry_mismatch` porque fue congelada con 81.0558 mm y el estéreo vigente usa 77.592459 mm.
- Se conserva una copia histórica en `sistema/calibracion_plataforma/calibracion_plataforma_PRE_CORRECCION_20260918.json`.
- Debe repetirse la calibración de plataforma con los pasos 07–09 corregidos antes de una reconstrucción nueva.

### Paso 10 — registro runtime
- Se exige la calibración estéreo vigente además de la calibración de plataforma.
- Se validan hashes, baseline, focal rectificada y resolución antes de registrar vistas.
- Una calibración antigua sin contrato estéreo se rechaza.
- El cierre P24->P00 ahora tiene gates duros adicionales:
  - overlap mínimo absoluto por defecto 0.70 cuando el cierre es informativo;
  - relación mínima cierre/mediana de pares primarios 0.75.
- Esto evita propagar un cierre degradado como simple warning.

### Paso 11 — fusión multivista
- La recuperación de cobertura observada se amplió conservadoramente, sin crear puntos sintéticos.
- Se permite recuperar más surfels observados si tienen soporte multivista, normal compatible, incertidumbre acotada y conexión con superficie retenida.
- Se añadieron gates duros de cobertura de muestreo antes del mallado:
  - retención mínima de celdas ocupadas: 35%;
  - cobertura mínima de candidatos a <=2 voxels: 80%.
- Si la nube queda demasiado hueca, el pipeline se detiene en 11 y no delega a Poisson el relleno de grandes regiones.

### Paso 13 — reconstrucción de superficie
- Poisson dejó de ser el ganador por defecto.
- Se añadió un gate duro de fidelidad observacional relativo al spacing para todos los candidatos.
- Si Poisson extrapola y BPA es geométricamente válido, BPA puede ganar aunque Poisson sea más continuo.
- Si todos los candidatos fallan fidelidad/cobertura/conectividad, el paso se rechaza en vez de seleccionar “el menos malo”.
- El trimming de densidad Poisson pasa de 0 a 1% por defecto como recorte conservador previo a los gates.

### Paso 17 — validación final
- La coherencia topológica ya no puede anular un rechazo mesh->cloud.
- `--allow-coherent-interpolation` queda desactivado por defecto y, aunque se active para diagnóstico, un P90 por encima del gate duro sigue siendo rechazo.

## Verificaciones realizadas
- `py_compile` sobre los 30 archivos Python del proyecto: 0 errores.
- `git diff --check`: sin errores de whitespace/parches.
- Contrato estéreo comprobado directamente:
  - `||T|| = 77.5924590721 mm`
  - baseline derivada de P2 = 77.5924590721 mm
  - baseline derivada de Q = 77.5924590721 mm
  - `fx = 2096.54995093 px`
  - mapas de rectificación = 1080x1920
- Prueba unitaria del letterbox CREStereo:
  - 1920x1080 -> contenido 640x360 dentro de 640x480;
  - padding vertical 60 px;
  - escala X = escala Y = 1/3.
- Prueba unitaria del selector del Paso 13:
  - Poisson extrapolado + BPA fiel -> selecciona BPA;
  - Poisson extrapolado + BPA extrapolado -> rechazo duro.
- La calibración de plataforma antigua es rechazada correctamente por estado/contrato.

## Limitación de la revisión en este entorno
El contenedor de revisión no dispone de `onnxruntime` ni `open3d`; por ello no se ejecutó una inferencia CREStereo real ni una reconstrucción Open3D completa. Sí se verificó sintaxis, contratos geométricos, funciones puras, selección de candidatos y preprocesamiento mediante mocks controlados. El proyecto incluye `onnxruntime-gpu` y `open3d>=0.19.0` en `requirements.txt`, por lo que la prueba end-to-end debe hacerse en el entorno `tesis` del equipo de captura.

## Secuencia recomendada para la siguiente prueba
1. Recalibrar la plataforma con la calibración estéreo vigente usando el modo de calibración del sistema.
2. Ejecutar una campaña completa desde cero; no reutilizar salidas 02–17 anteriores.
3. Revisar primero los resúmenes 02, 04, 05, 10 y 11.
4. Solo permitir que 13 se ejecute si 11 supera sus nuevos gates de cobertura.
5. Verificar que 17 finalice como `accepted` o, como máximo, warnings que no contradigan ningún gate duro de fidelidad.
