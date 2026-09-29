# Galería de resultados 3D

Sitio estático para GitHub Pages. Carga bajo demanda los PLY originales, sin simplificación ni modificación de los archivos científicos. Centra la geometría y gira 180° alrededor de X solo en memoria para mostrar hacia arriba el eje vertical que los archivos expresan hacia abajo. Las descargas conservan coordenadas y unidades originales. La versión previa al pulido puede incluir relleno; la vista de vértices no es una nube observada independiente.

## Vista previa local

Desde la raíz, con Python (sin dependencias adicionales):

```powershell
python herramientas/preparar_visor_web.py
python -m http.server 8000 --directory _site
```

Abrir http://localhost:8000. No abrir el HTML con doble clic: los módulos y los datos necesitan HTTP. `_site/` es salida temporal ignorada por Git y se regenera por completo.

## Publicar en GitHub Pages

1. Subir los cambios de esta implementación a `main`, incluidos `visor/`, el script y `.github/workflows/pages.yml`.
2. En GitHub: **Settings → Pages → Build and deployment → Source → GitHub Actions**.
3. En **Actions → Publicar visor 3D → Run workflow**, ejecutar sobre `main` si el primer despliegue no se inició después de activar Pages.
4. Al finalizar correctamente, abrir https://mategut.github.io/Sistema_3D/.

Los siguientes cambios en el visor o en resultados publican de nuevo automáticamente. La URL no se considera activa hasta que el despliegue termine. No se necesita dominio propio, servidor Python ni subir `trabajos/` o el modelo ONNX. El artefacto contiene exclusivamente el visor y la lista explícita de resultados públicos del script.

## Contenido y límites

- Selección de nueve campañas y variantes final/anterior al pulido; enlaces directos mediante `#cubo1`, por ejemplo.
- Superficie, triángulos y vértices, giro automático opcional, centrar y controles táctiles/teclado.
- Métricas originales del resultado final (no se recalculan al alternar variante); dimensiones exploratorias de la variante seleccionada; advertencias e imágenes originales.
- Requiere un navegador moderno con módulos JavaScript y WebGL2. Si falla el contexto gráfico, siguen disponibles las imágenes y descargas. Los archivos grandes consumen memoria y ancho de banda: solo se mantiene un modelo a la vez.
- No modifica el programa de adquisición, los informes, las huellas ni los PLY. No ejecuta validaciones científicas nuevas. No permite medir distancias haciendo clic.

## Dependencia del visor

Three.js **r180 / 0.180.0**, distribuido localmente en `vendor/` desde el repositorio oficial https://github.com/mrdoob/three.js/tree/r180. Incluye `three.module.js`, `three.core.js`, `OrbitControls.js`, `PLYLoader.js` y su licencia MIT en `vendor/LICENSE`. No depende de un CDN en tiempo de ejecución. Esta atribución corresponde a la biblioteca. El código propio del visor y su documentación se distribuyen bajo la [licencia MIT del proyecto](../LICENSE); las licencias y avisos de Three.js se conservan por separado.

El catálogo y las métricas se leen directamente de los JSON existentes. Para añadir campañas, actualizar primero la publicación científica en `resultados/` y luego regenerar el sitio.
