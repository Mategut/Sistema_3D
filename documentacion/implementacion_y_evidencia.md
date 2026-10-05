# Implementación y evidencia del sistema

[Volver al README](../README.md)

## Alcance de la evaluación

La caracterización combina dimensiones de objetos de referencia, indicadores de alineación multivista, fidelidad nube–malla y topología. El tercer objetivo del documento académico es:

> Evaluar la precisión y la calidad geométrica del modelo tridimensional obtenido mediante la comparación dimensional con objetos poligonales de referencia y el análisis de indicadores de alineación multivista, fidelidad geométrica y topología.

Los objetivos de captura controlada y desarrollo de reconstrucción métrica se mantienen. Los indicadores describen la estrategia implementada; no se declara superioridad frente a ICP libre u otras estrategias.

## Geometría del montaje

La base mide aproximadamente 30 × 50 cm. Los 40 cm son desde el borde frontal hasta el centro de la plataforma. Las cámaras están retranqueadas unos 5 cm; su distancia longitudinal nominal al centro es de unos 35 cm. La separación entre centros ópticos es aproximadamente 7,7 cm.

La calibración histórica conserva 355,196 mm como distancia del punto medio estéreo a la recta del eje. Es un descriptor tridimensional distinto de la cota longitudinal nominal. El informe original utilizó un prior de 400 mm con tolerancia amplia y se conserva intacto. Los valores por defecto de los pasos 08/09 para nuevas calibraciones son 350 mm. No se modifican calibraciones activas, referencias congeladas ni mallas históricas.

## Interfaz y operación

Las imágenes muestran los controles reales en estado inicial, sin dispositivos conectados; las vistas oscuras no son capturas experimentales.

![Pestaña captura](imagenes/interfaz_captura.png)

![Pestaña calibracion](imagenes/interfaz_calibracion.png)

![Pestaña herramientas](imagenes/interfaz_herramientas.png)

Captura reúne conexión, confirmación de cámaras, fondo vacío, creación de campañas y adquisición de tres sesiones. Calibración permite crear/importar/verificar estéreo y crear la campaña de plataforma. «Calibrar plataforma» y «Crear campaña de calibración» invocan la misma acción. Herramientas reúne diagnóstico LR, registros, motor de inferencia y almacenamiento.

«Procesar desde cero» limpia los productos previos del trabajo. «Reanudar procesamiento» reutiliza checkpoints válidos y recalcula desde la primera etapa afectada. No equivale a continuar una captura interrumpida: la posición inicial debe restablecerse físicamente.

## Contratos de procesamiento

La [arquitectura y productos por etapa](arquitectura_y_codigo.md) identifica cada script.

| Grupo | Entradas | Operación y salidas | Control |
| --- | --- | --- | --- |
| 01–04 | Capturas, ONNX, estéreo, fondo | Mapa angular, profundidad, máscaras y selección | Correspondencia, LR, dominio y procedencia |
| 05–06 | Observaciones de igual pose | Consenso y nubes métricas con atributos | Acuerdo y respaldo de sesiones |
| 07–09 | Cubo de referencia | Geometría, eje, candidata y auditoría | Registro, cierre y contrato estéreo |
| 10 | Nubes y calibración | Registro y evidencia multivista | Alineación, cierre y refinamiento condicionado |
| 11–12 | Vistas registradas | Fusión y regularización | Soporte angular, confianza, conflicto y desplazamiento |
| 13–15 | Nube y candidatos | Superficie, topología y pulido | Fidelidad, cobertura, conectividad y controles locales |
| 16–18 | Malla y evidencia | Intersecciones, calidad y exportación | Estado, huella y unidades |

## Consenso y profundidad absoluta

El paso 05 realiza alineación bidimensional restringida y selecciona una capa compatible mediante medoide. Combina observaciones en profundidad inversa con confianza, residual y ponderación robusta. Conserva dispersión y fiabilidad. Las diferencias escalares de profundidad se registran como diagnóstico; la estimación devuelve correcciones nulas, sin modificar la profundidad absoluta para igualar sesiones. El soporte entre sesiones de una pose no equivale a soporte angular de poses distintas.

## Registro, fusión y superficie

El registro aplica la referencia de plataforma; el refinamiento transversal conserva dirección y ángulos y requiere controles independientes. La fusión utiliza soporte, incidencia, confianza, dispersión y coherencia local en un marco canónico con Y alineado al eje. El paso 13 exige fidelidad, cobertura y conectividad antes de comparar puntuaciones. Si todos los candidatos fallan, rechaza la etapa. Una superficie estimada o continua no constituye observación íntegra ni verdad de referencia.

Los nueve informes conservan `runtime_axis_line_refinement_used = false`. Los cilindros 1 y 3 conservan Poisson; los otros siete resultados conservan paredes continuas con base abierta. El caso de cubo 1 seleccionó inicialmente Poisson y aceptó después el completado; su reporte identifica todas las caras finales como estimadas.

| Campaña | RMSE primario mediano (mm) | RMSE cierre punto–plano (mm) | Método final | Paso 13 interno (s) |
| --- | ---: | ---: | --- | ---: |
| cilindro1 | 0.307 | 0.365 | screened_poisson_adaptive | 119.0 |
| cilindro2 | 0.298 | 0.273 | continuous_walls_open_base | 104.0 |
| cilindro3 | 0.289 | 0.351 | screened_poisson_adaptive | 65.2 |
| cubo1 | 0.442 | 1.099 | continuous_walls_open_base | 316.2 |
| cubo2 | 0.361 | 0.491 | continuous_walls_open_base | 707.2 |
| cubo3 | 0.383 | 0.623 | continuous_walls_open_base | 367.1 |
| piramide1 | 0.625 | 0.518 | continuous_walls_open_base | 34.3 |
| piramide3 | 0.362 | 0.391 | continuous_walls_open_base | 31.2 |
| piramide4 | 0.369 | 0.289 | continuous_walls_open_base | 73.3 |

## Parámetros decisivos

| Grupo | Valor | Procedencia | Efecto |
| --- | --- | --- | --- |
| Estéreo | Tablero 6 por 8; cuadro 23 mm | Reporte disponible | Determina escala y geometría. |
| Montaje | 350 mm cámaras–eje nominal | Valor actual de pasos 08/09 | 40 cm desde borde menos 5 cm de retranqueo. |
| Consenso 05 | Acuerdo 6 mm; soporte mínimo 2 | Informe de cubo 1 | Selecciona profundidad compatible. |
| Fusión 11 | Vóxel 1.5 mm; poses independientes 2 | Informe de cubo 1 | Escala local y respaldo angular. |
| Regularización 12 | SOR: 24 vecinos y razón 2.5 | Informe de cubo 1 | Retirada de puntos aislados. |
| Superficie 13 | Cobertura mínima 0.90 | Selección registrada en cubo 1 | Admisibilidad interna de candidato. |
| Validación 17 | Radio de cobertura 3 mm | Informe histórico | Fracción de la nube próxima a malla. |
| Exportación 18 | Factor 0.001 y permutación de ejes | Código actual | Milímetros a metros para Blender. |

Los valores de cubo 1 no se atribuyen automáticamente a todas las campañas. La [referencia completa](referencia_parametros.md) recoge valores declarados del código y los informes identifican valores efectivos.

## Calibración de plataforma registrada

Las 25 relaciones primarias aprobaron; mediana RMSE punto–plano 0,506 mm; mediana P90 0,740 mm; solapamiento de cierre 0,983; RMSE robusto de cierre 1,085 mm y punto–plano de cierre 0,776 mm. Son indicadores internos del cubo de referencia, sin certificación metrológica independiente.

## Montaje construido

![Montaje real general](imagenes/montaje_real_general.jpeg)

![Interior del montaje](imagenes/montaje_real_interior.jpeg)

Fotografías suministradas del montaje; no se utilizan para deducir dimensiones por perspectiva.

## Secuencia visual conservada de cubo 1

![Siluetas, paso 03](imagenes/etapa_mascara.png)

![Máscaras de validación, paso 04](imagenes/etapa_validacion.png)

![Nubes por pose, paso 06](imagenes/etapa_nubes.png)

![Registro, paso 10](imagenes/etapa_registro.png)

![Fusión, paso 11](imagenes/etapa_fusion.png)

![Pulido, paso 15](imagenes/etapa_pulido.png)

Son diagnósticos originales de cubo 1: las hojas de contacto abarcan varias poses y los previews geométricos son agregados. No se reconstruyen mapas individuales retirados por compactación; la máscara validada no es una escala de distancia.

## Rendimiento histórico identificado

El registro `Piramide3_20260922_211146_20260923_104137.log` solicita reanudación, pero invalida desde 01 y ejecuta las 21 invocaciones de la ruta normal. Sus tiempos corresponden con 21 archivos de telemetría de salida cero. La suma externa de etapas es **4672,1 s (77,9 min)**, sin adquisición ni todos los intervalos entre procesos.

| Paso | Operación | Tiempo externo (s) |
| --- | --- | ---: |
| 01 | Mapa angular | 5.4 |
| 02 | Profundidad, tres sesiones | 575.7 |
| 03 | Máscaras, tres sesiones | 238.3 |
| 04 | Validación, tres sesiones | 1962.1 |
| 05 | Consenso | 42.8 |
| 06 | Nubes | 120.6 |
| 10 | Registro | 62.9 |
| 11 | Fusión | 923.5 |
| 12 | Regularización | 6.4 |
| 13 | Superficie | 33.1 |
| 14 | Topología | 118.8 |
| 15 | Pulido | 482.1 |
| 16 | Intersecciones | 62.1 |
| 17 | Validación final | 35.5 |
| 18 | Exportación | 2.8 |

![Tiempos por etapa](imagenes/rendimiento_etapas.png)

Las etapas 02–04 agrupan tres sesiones. Las muestras de CPU, RAM disponible y GPU son del equipo completo; no acreditan consumo exclusivo ni proveedor ONNX. Los tiempos internos de superficie tienen otro alcance y no se suman al registro externo.

## Procedencia y alcance de las fuentes

El [resumen de evidencia](evidencia_implementacion/resumen_implementacion.json) conserva indicadores extraídos, identificadores relativos y huellas SHA-256 de los originales. Las fuentes permanecen en las campañas locales; no se modifican métricas ni mallas. El JSON incluye las 21 series de telemetría utilizadas y la suma de tiempos.

La formulación del anteproyecto se materializa en adquisición, calibración, nubes, eje, integración y evaluación; la aplicación añade consenso, campañas, referencias congeladas y publicación. Presupuesto y cronograma de planeación no equivalen a gastos finales o fechas ejecutadas. La comparación de estrategias alternativas queda fuera de la evaluación descrita.
