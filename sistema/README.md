# Referencias del montaje

[Volver al proyecto](../README.md)

Al preparar una campaña, la aplicación utiliza las referencias de esta carpeta. Las incluidas corresponden al montaje de desarrollo. Antes de emplearlas, compruebe la disposición de cámaras y plataforma; si el montaje es diferente, obtenga referencias propias.

## Contenido

| Carpeta | Archivos principales y función |
| --- | --- |
| `calibracion_estereo/` | `stereo_initial.yaml`: matrices y parámetros estéreo; `rectification_maps.npz`: mapas de rectificación; `calibration_report.json`: informe de la calibración. |
| `calibracion_plataforma/` | `calibracion_plataforma.json`: referencia instalada; `calibracion_plataforma_candidata.json`: candidata; `auditoria_congelacion_calibracion.json`: controles y huella; `transformaciones_mecanicas_25_poses.csv`: transformaciones del montaje. |
| `fondo_vacio/` | `background_left.png` y `background_right.png`: imágenes de fondo; `background_pair.png`: vista del par; `background_metadata.json`: metadatos de adquisición. |

La aplicación también puede generar `estructura_sistema.json`. Es un archivo local excluido de Git, no una referencia que deba publicarse.

## Candidata, referencia instalada y activación

El paso 09 genera la calibración dentro de la campaña. Si sus controles se aprueban, la aplicación la instala automáticamente en esta carpeta, comprueba su correspondencia estéreo y conserva un respaldo de la referencia anterior.

La instalación habitual registra `active_step09_approved` en `estado_activacion_plataforma.json` con la huella correspondiente. **Instalar calibración guardada…** realiza la misma operación sobre un resultado existente. No necesita protocolo ni informe del paso 17. Los estados históricos `evaluation_only` y `active_independently_validated` siguen describiendo sus modalidades de evaluación y revisión; no se reetiquetan automáticamente.

El archivo `calibracion_plataforma.json` contiene la referencia instalada. Para conocer su modalidad de aprobación, consulte los estados y la evidencia asociada: el nombre del archivo no acredita una revisión independiente, y la validación estructural no certifica exactitud dimensional.

El procedimiento completo está en la [guía de calibración](../documentacion/validacion_independiente_plataforma.md).

## Cuándo actualizar las referencias

- **Estéreo:** recalibre si cambia la posición relativa de las cámaras, su orientación o una configuración óptica que afecte la calibración. Utilice la resolución prevista por el sistema.
- **Plataforma:** vuelva a calibrar y evaluar si cambia su geometría respecto a las cámaras o si sustituye la calibración estéreo. La importación de una nueva referencia estéreo archiva la referencia de plataforma anterior.
- **Fondo vacío:** capture de nuevo cuando cambien el fondo, la iluminación o los ajustes de captura. Hágalo sin el objeto y con el montaje estabilizado.

Realice estas operaciones desde la interfaz siguiendo [instalación y uso](../documentacion/instalacion_y_uso.md#calibraciones-desde-la-interfaz). Los mecanismos de importación y promoción conservan respaldos locales en `registros/`.

## Referencias de cada campaña

Al crear una campaña, el sistema conserva copias de sus referencias en `trabajos/<campaña>/documentacion/referencias/`. Esas copias quedan congeladas para el procesamiento, que comprueba su integridad. Los cambios posteriores en la carpeta global no se trasladan a campañas ya existentes.

Conserve las referencias originales de las capturas. Si el montaje cambia, prepare una nueva campaña con referencias correspondientes al nuevo montaje. Las campañas antiguas sin copias congeladas tienen un tratamiento de compatibilidad explicado en las guías; no ofrecen la misma trazabilidad.

Los recursos se identifican mediante SHA-256 y Git conserva sus bytes mediante `.gitattributes`. Si una comprobación falla, revise o regenere la referencia correspondiente; modificar matrices, mapas o huellas manualmente invalida su trazabilidad.
