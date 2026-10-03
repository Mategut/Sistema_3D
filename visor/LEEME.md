# Visor de resultados 3D

[**Abrir el visor interactivo**](https://mategut.github.io/Sistema_3D/)

El visor permite consultar las nueve campañas publicadas del proyecto Sistema 3D desde el navegador, sin instalar la aplicación de reconstrucción. El contenido publicado corresponde a la selección oficial de resultados del repositorio. Los visitantes pueden explorar y descargar los resultados; la página no ofrece funciones para subir archivos ni modificar las campañas publicadas.

## Cómo utilizarlo

1. Seleccionar una campaña: cilindros 1, 2 y 3; cubos 1, 2 y 3; o pirámides 1, 3 y 4.
2. Elegir el modelo **Final** o **Antes del pulido**.
3. Arrastrar para girar y utilizar la rueda del ratón o un gesto de pellizco para acercar o alejar. El botón derecho permite desplazar la vista.
4. Cambiar entre **Superficie**, **Triángulos** y **Vértices de la malla**. **Centrar** recupera el encuadre y **Giro automático** activa o detiene la rotación.
5. Consultar las métricas, dimensiones, advertencias e imágenes debajo del modelo. El enlace **Descargar PLY mostrado** descarga la variante seleccionada.

Con el área del visor enfocada, las flechas giran la vista, `+` y `−` ajustan el acercamiento y `0` centra el objeto. Se puede compartir una campaña mediante un enlace directo, por ejemplo [Cubo 1](https://mategut.github.io/Sistema_3D/#cubo1).

## Cómo interpretar los resultados

- Las métricas de fidelidad interna corresponden al resultado final, incluso cuando se muestra la malla anterior al pulido. La tabla dimensional sí corresponde a la variante seleccionada.
- Las referencias físicas fueron tomadas con regla y son aproximadas. En la pirámide, los 85 mm corresponden a la altura de cara, no a la altura perpendicular.
- Los estados de aceptación y las advertencias son controles internos del sistema; no constituyen una certificación de exactitud dimensional.
- La malla anterior al pulido también puede contener superficies estimadas. La vista de vértices representa puntos de la malla, no la nube observada original.
- Los PLY se cargan sin simplificar. Se centran y giran solo en pantalla para facilitar su inspección; las descargas conservan las coordenadas y unidades originales en milímetros.

El visor requiere un navegador moderno con WebGL2. Carga un modelo a la vez; el tiempo de apertura depende de la conexión y del equipo. Si la visualización 3D no está disponible, se pueden consultar las imágenes y descargar los archivos. No realiza nuevas reconstrucciones ni permite medir distancias haciendo clic.

## Mantenimiento del sitio oficial

Las actualizaciones del sitio se gestionan desde el repositorio oficial. GitHub Pages aloja la página; los visitantes no necesitan activar servicios ni ejecutar comandos.

El flujo [Publicar visor 3D](../.github/workflows/pages.yml) prepara y despliega el sitio cuando se suben cambios pertinentes a `main`. Utiliza [preparar_visor_web.py](../herramientas/preparar_visor_web.py) para reunir el visor y los resultados públicos en `_site/`, una carpeta generada que no se incorpora al repositorio. Las campañas de trabajo y el modelo ONNX no forman parte del sitio publicado.

## Biblioteca y licencia

El renderizado utiliza [Three.js r180 / 0.180.0](https://github.com/mrdoob/three.js/tree/r180), incluida localmente en `vendor/` con su [licencia MIT](vendor/LICENSE). El código propio del visor y su documentación se distribuyen bajo la [licencia MIT del proyecto](../LICENSE), con el alcance indicado en el [README principal](../README.md#licencia).

La reutilización de copias del código conforme a su licencia no concede permisos para modificar este repositorio ni su sitio oficial.
