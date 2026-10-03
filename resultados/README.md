# Resultados seleccionados

La selección publicada comprende cilindros 1, 2 y 3; cubos 1, 2 y 3; pirámides 1, 3 y 4. Se mantienen las calidades y advertencias históricas. `cubo3` corresponde a la carpeta original `ubo3_20260919_131506`, cuyo nombre se escribió sin la C. Las campañas originales no se renombran.

- [cilindro1](cilindro1/README.md): `cilindro1_20260919_132249`
- [cilindro2](cilindro2/README.md): `cilindro2_20260919_132519`
- [cilindro3](cilindro3/README.md): `cilindro3_20260919_132734`
- [cubo1](cubo1/README.md): `cubo1_20260919_130526`
- [cubo2](cubo2/README.md): `cubo2_20260919_131206`
- [cubo3](cubo3/README.md): `ubo3_20260919_131506`
- [piramide1](piramide1/README.md): `piramide1_20260920_183454`
- [piramide3](piramide3/README.md): `Piramide3_20260922_211146`
- [piramide4](piramide4/README.md): `piramide4_20260922_175555`

Consulte las [medidas físicas aproximadas](medidas_fisicas.md) y su [registro JSON](medidas_fisicas.json). Las nueve campañas se asocian a las referencias por tipo de objeto.

## Uso y reproducción

Abra los PLY con un visor compatible. Sus coordenadas están en milímetros: para trabajar en metros aplique escala 0.001. Las imágenes son evidencia ilustrativa, no las 75 parejas de una campaña. Las mallas no se simplificaron ni se recalcularon al preparar este paquete.

Para repetir el procesamiento se necesitan las capturas completas S01/S02/S03, sus referencias congeladas y el modelo correspondiente, que permanecen en trabajos y no se duplican aquí. Abra la campaña original desde la aplicación y procese con conservación completa; un cambio de código puede modificar los resultados. Este paquete permite inspección de los resultados publicados, no reproducción numérica completa por sí solo.

Para volver a generar esta selección local: `python herramientas/preparar_resultados_publicos.py --output resultados_nuevos`. Consulte `procedencia.json` en cada ejemplo y `catalogo.json`. Las rutas personales se omiten de los informes publicados; por ello sus huellas difieren de los originales. Las referencias físicas de la comparación están documentadas en [medidas_fisicas.md](medidas_fisicas.md).

`MANIFEST.json` registra los bytes de todos los archivos del paquete excepto él mismo. Git conserva estos archivos sin conversión de finales de línea.

## Comparación dimensional

[Tabla de comparación con regla](comparacion_dimensional.md), con resultados finales y previos al pulido, métodos, discrepancias y variación entre campañas.
