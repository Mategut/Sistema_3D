# Calibración de plataforma

[Volver a instalación y uso](instalacion_y_uso.md)

## Uso habitual desde la interfaz

1. Prepare la calibración estéreo y el fondo vacío correspondientes al montaje.
2. Pulse **Crear campaña de calibración** en Calibración, o **Calibrar plataforma** en Captura. Ambos preparan el mismo trabajo.
3. Coloque un **cubo como objeto de referencia**, capture S01, S02 y S03 y ejecute el procesamiento.
4. La ruta de calibración termina en los pasos 07, 08 y 09. Si sus controles se aprueban, la aplicación instala automáticamente la calibración y muestra **Calibración de plataforma instalada y lista para usar**.
5. Cree una nueva campaña de objeto y reconstruya normalmente.

No necesita escribir un protocolo, preparar un informe externo ni obtener el paso 17 de la campaña de calibración. El paso 17 pertenece a las reconstrucciones de objetos.

## Cubo de referencia y colocación

Utilice un cubo con caras planas y perpendiculares entre sí: el flujo de calibración aplica comprobaciones geométricas de cuboide.

- Capture el fondo vacío antes de colocar el cubo.
- Apoye una cara completa sobre la plataforma, de forma estable, y compruebe que el cubo permanezca dentro de la vista de ambas cámaras durante toda la vuelta.
- Mantenga el cubo fijo respecto a la plataforma durante las tres sesiones; no lo levante ni lo reposicione entre ellas.
- Mantenga fijas las cámaras y la iluminación. Evite superficies brillantes o transparentes que dificulten observar las caras.

Una vez instalada la calibración aprobada por el paso 09, puede retirar el cubo y crear una nueva campaña para reconstruir otro objeto.

## Qué se comprueba y conserva

No basta con que exista un archivo del paso 09: la instalación exige `candidate_quality_passed=true`, auditoría aprobada con la huella de esa calibración y correspondencia con la referencia estéreo actual. Si falla una comprobación, no se sustituye la referencia del sistema.

El resultado original permanece en `trabajos/<campaña>/resultado_calibracion_plataforma/`. La aplicación instala una copia en `sistema/calibracion_plataforma/` y conserva la referencia anterior en `registros/calibracion_plataforma_reemplazada_<marca_temporal>/`. Las campañas ya creadas mantienen sus referencias congeladas.

La aprobación operativa se registra como `active_step09_approved` en `estado_activacion_plataforma.json`. El JSON geométrico conserva sus bytes y su nombre de candidata para mantener las huellas. Aprobar los controles del paso 09 no equivale a certificar exactitud dimensional ni generalización experimental.

## Instalar un resultado guardado

Si ya dispone de una calibración calculada, pulse **Instalar calibración guardada…** y seleccione `calibracion_plataforma_candidata.json` en su carpeta de resultados. Se comprueban los mismos requisitos y se respalda la referencia anterior. Esto también permite reintentar una instalación que falló después de calcular la calibración.

Si utiliza únicamente los scripts de consola, el coordinador guarda el resultado en la carpeta indicada; puede instalarlo desde ese botón o con:

```powershell
python herramientas/promover_calibracion_plataforma.py activar --candidate-dir ".\trabajos\CALIBRACION_NUEVA\resultado_calibracion_plataforma"
```

Cierre la aplicación si utiliza la instalación desde consola y ábrala de nuevo después.

## Modalidades de aprobación

La instalación automática utiliza la aprobación operativa del paso 09 y registra `active_step09_approved`. Las herramientas de consola también admiten informes de revisión mediante `--evidence` y el modo `evaluar`. El estado `active_independently_validated` identifica la modalidad con evidencia independiente; no describe la instalación automática ni constituye un requisito para utilizar el flujo habitual.
