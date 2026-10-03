# Método e interpretación de las mediciones dimensionales

[Resultados y validación](resultados_y_validacion.md) · [Comparación publicada](../resultados/comparacion_dimensional.md)

El [script de comparación](../herramientas/comparar_dimensiones.py) utiliza coordenadas en milímetros, sin reescalar las mallas a las medidas físicas. Las diferencias se calculan como reconstrucción menos referencia y el porcentaje se divide entre la referencia. La referencia de 85 mm de la pirámide corresponde a la altura de cara, no a la altura perpendicular al plano de base. Este documento describe el método utilizado y la interpretación de las discrepancias de la muestra publicada.

## Sensibilidad de las estimaciones

| Objeto | Método actual | Cómo puede influir en la diferencia |
| --- | --- | --- |
| Cubo | Altura por extremos Y y base por rectángulo envolvente mínimo en XZ | Salientes e inclinación pueden aumentar la envolvente. Ordenar lado menor como fondo y mayor como ancho no identifica las caras físicas; las referencias difieren solo 1 mm. |
| Cilindro | Altura por extremos Y; diámetro por ajuste circular robusto en la franja central del 20–80 % | Los extremos afectan la altura. La inclinación del eje y la ponderación por densidad de vértices pueden afectar el diámetro; el ajuste robusto reduce la influencia de salientes, pero no certifica la escala. |
| Pirámide triangular | Base ajustada en una banda de 2 mm, esquinas de su envolvente y punta promediada en la banda opuesta | El redondeo y el promedio de la punta pueden reducir longitudes. Se comparan medias de tres lados, aristas y alturas de cara; medir un único segmento físico no equivale necesariamente a esa media. |

## Diagnósticos de las mallas publicadas

Los diagnósticos de las mallas finales permiten comparar la altura por extremos con la extensión entre percentiles 0,5–99,5, a partir de los registros incluidos en la entrega:

| Campaña | Altura por extremos (mm) | Extensión entre percentiles (mm) | Diferencia entre métodos (mm) |
| --- | ---: | ---: | ---: |
| cilindro1 | 99,178 | 97,278 | 1,900 |
| cilindro2 | 97,955 | 97,070 | 0,886 |
| cilindro3 | 99,022 | 97,193 | 1,829 |
| cubo1 | 95,465 | 95,423 | 0,042 |
| cubo2 | 95,848 | 95,816 | 0,032 |
| cubo3 | 95,977 | 95,470 | 0,507 |

Fuente: `diagnostics.height_robust_005_995_mm` y `dimensions_mm` del [JSON publicado](../resultados/comparacion_dimensional.json). La extensión entre percentiles elimina parte de los extremos por definición; no sustituye la altura física ni demuestra que dichos extremos sean erróneos. La tabla muestra sensibilidad especialmente apreciable en las alturas de cilindro.

Los diámetros de cilindro publicados se obtuvieron mediante ajuste robusto, no mediante una caja envolvente. Por ello, las diferencias positivas de diámetro no pueden atribuirse simplemente a que se eligió el punto más lejano. Los diagnósticos publicados no separan cuantitativamente los efectos de calibración, captura, reconstrucción, estimación de superficie y referencia física.

## Alcance de la comparación

La comparación caracteriza las discrepancias de las nueve campañas publicadas respecto de referencias aproximadas tomadas con regla. La sensibilidad de los descriptores geométricos puede contribuir a esas diferencias, pero los datos disponibles no permiten atribuirles una causa única o principal ni cuantificar la incertidumbre de la medición física.

Las referencias físicas de la pirámide describen segmentos individuales; los descriptores de la malla corresponden a medias de tres lados, aristas y alturas de cara. Esta diferencia de definición forma parte de las limitaciones de la comparación. La altura de cara es el segmento sobre la cara desde la punta perpendicularmente al lado de base; las medidas físicas proceden de la regla, no de longitudes en píxeles de una fotografía en perspectiva.
