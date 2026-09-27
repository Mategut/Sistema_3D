# Validación independiente de la plataforma

[Volver a instalación y uso](instalacion_y_uso.md)

El paso 09 genera una **candidata**. La aplicación la conserva dentro de la campaña y no reemplaza automáticamente la referencia del sistema. Una reconstrucción aceptada internamente no demuestra por sí sola generalización ni exactitud dimensional absoluta.

## Flujo desde la interfaz (sin PowerShell)

En la pestaña **Calibración**, use **Usar candidata para evaluación** y seleccione `calibracion_plataforma_candidata.json` dentro de `resultado_calibracion_plataforma`. Se comprobarán la auditoría y la correspondencia estéreo antes de instalarla para pruebas.

Después de adquirir y procesar las campañas independientes, abra **Preparar informe de revisión**. Seleccione la candidata y el protocolo experimental, escriba el nombre del responsable y añada cada campaña con su geometría. Declare explícitamente la independencia de adquisición, si la geometría no fue usada en el ajuste y si revisó las advertencias. Todas las casillas comienzan desmarcadas. El formulario calcula las huellas SHA-256; no necesita editarlas a mano.

**Guardar y verificar informe** guarda el JSON y comprueba sus archivos en segundo plano. Si falla, el informe queda guardado pero **no validado**; corrija las evidencias o las declaraciones según el diagnóstico. Guardarlo o verificarlo no activa la calibración. Cuando pase, use **Activar con informe revisado** y seleccione la misma candidata y el informe. La activación vuelve a verificar todo y archiva la referencia anterior.

Los informes creados por el formulario usan rutas absolutas locales. Si traslada las evidencias a otro equipo, actualice las rutas o prepare un informe nuevo allí. La independencia física y las conclusiones experimentales siguen requiriendo revisión humana; únicamente las comprobaciones de archivos se automatizan.

## Evaluar una candidata

Como alternativa, puede usar la línea de comandos. Con cámaras y plataforma en el montaje correspondiente, instale explícitamente la candidata para adquirir campañas de evaluación. Para esta alternativa cierre la aplicación durante el cambio y vuelva a abrirla después:

```powershell
python herramientas/promover_calibracion_plataforma.py evaluar `
  --candidate-dir ".\trabajos\CALIBRACION_NUEVA\resultado_calibracion_plataforma"
```

El comando verifica la auditoría de congelación y la correspondencia con la calibración estéreo activa. Conserva la referencia anterior en `registros/`, instala una copia operativa y registra `evaluation_only`. La interfaz muestra «Candidata operativa · validación pendiente». La geometría y los bytes de la candidata no se modifican.

Cree al menos tres campañas nuevas e independientes, que incluyan al menos dos geometrías no vistas durante el ajuste de calibración. S01/S02/S03 son sesiones de una sola campaña. Mantenga fija la candidata y los parámetros de procesamiento, y conserve las capturas y referencias congeladas. No sustituya referencias de campañas ya capturadas para aparentar que usaron otra calibración.

## Informe de revisión

El responsable debe revisar la independencia física de las adquisiciones, la clasificación de geometrías, las advertencias y la ausencia de ajuste de parámetros por objeto. Estos hechos requieren revisión experimental; el software solo puede comprobar archivos y declaraciones. Registre el procedimiento, parámetros, mediciones externas y conclusiones en un protocolo de texto.

Prepare un JSON como el siguiente. Los valores de ejemplo son marcadores: no son evidencia ni pasan las comprobaciones. Las rutas relativas se resuelven respecto a la carpeta de este JSON.

```json
{
  "schema_version": 1,
  "calibration_sha256": "SHA256_DE_LA_CANDIDATA",
  "reviewer": "Nombre del responsable",
  "fixed_parameters_reviewed": true,
  "protocol_file": "protocolo_validacion.txt",
  "protocol_sha256": "SHA256_DEL_PROTOCOLO",
  "campaigns": [
    {
      "workspace": "../trabajos/EVALUACION_01",
      "geometry": "geometria_A",
      "unseen_geometry": true,
      "independent_acquisition": true,
      "validation_sha256": "SHA256_DEL_RESUMEN_17_DE_ESTA_CAMPANA",
      "warnings_reviewed": true
    },
    {
      "workspace": "../trabajos/EVALUACION_02",
      "geometry": "geometria_B",
      "unseen_geometry": true,
      "independent_acquisition": true,
      "validation_sha256": "SHA256_DEL_RESUMEN_17_DE_ESTA_CAMPANA",
      "warnings_reviewed": true
    },
    {
      "workspace": "../trabajos/EVALUACION_03",
      "geometry": "geometria_A",
      "unseen_geometry": true,
      "independent_acquisition": true,
      "validation_sha256": "SHA256_DEL_RESUMEN_17_DE_ESTA_CAMPANA",
      "warnings_reviewed": true
    }
  ]
}
```

Obtenga cada SHA-256 de los archivos reales, por ejemplo:

```powershell
(Get-FileHash -Algorithm SHA256 -LiteralPath ".\ruta\archivo.json").Hash.ToLowerInvariant()
```

No declare `true` hasta haber comprobado el criterio. Los informes del paso 17 están en `reconstruccion/multisesion/17_validacion_modelo/resumen_17_validacion_modelo.json` dentro de cada campaña. Una advertencia necesita revisión explícita; un resultado rechazado no sirve para la promoción.

## Activar con revisión independiente

```powershell
python herramientas/promover_calibracion_plataforma.py activar `
  --candidate-dir ".\trabajos\CALIBRACION_NUEVA\resultado_calibracion_plataforma" `
  --evidence ".\documentacion_local\validacion_independiente.json"
```

La herramienta comprueba el número de campañas y geometrías declaradas, huellas de calibración y validaciones, referencias congeladas, capturas duplicadas, advertencias revisadas y el protocolo. Rechaza la promoción si falta evidencia. Guarda una copia del informe y protocolo junto a `estado_activacion_plataforma.json`, y respalda la referencia anterior antes de reemplazarla.

El estado `active_independently_validated` se guarda en ese archivo separado. El `status` del JSON geométrico sigue describiendo su generación como candidata: conservarlo inmutable permite verificar que la calibración activada es exactamente la usada en la evaluación. La calibración histórica incluida en el repositorio no se reclasifica automáticamente.

Una revisión de plataforma no equivale a certificar la precisión absoluta de cada objeto. Esa conclusión requiere medidas físicas independientes y sus incertidumbres.
