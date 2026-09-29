import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { PLYLoader } from './vendor/PLYLoader.js';

const $ = id => document.getElementById(id);
const fmt = n => Number.isFinite(n) ? n.toLocaleString('es-CO', {maximumFractionDigits: 3}) : 'No disponible';
const title = name => name.replace('piramide', 'Pirámide ').replace('cilindro', 'Cilindro ').replace('cubo', 'Cubo ');
const filenames = {final: 'modelo_final_original_mm.ply', pre_pulido: 'modelo_pre_pulido_mm.ply'};
const labels = {diametro:'Diámetro',altura:'Altura',alto:'Alto',ancho:'Ancho',fondo:'Fondo',lado_base:'Lado de base',altura_cara:'Altura de cara',arista_lateral_hasta_punta:'Arista lateral'};
const stage = $('stage');
let renderer, controls, object, selected, request = 0, abort;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#e7eeeb');
const camera = new THREE.PerspectiveCamera(40, 1, .1, 10000);
scene.add(new THREE.HemisphereLight(0xffffff, 0x50605a, 2.5));
const light = new THREE.DirectionalLight(0xffffff, 3);
light.position.set(150, 250, 200); scene.add(light);
try {
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  stage.append(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener('change', render);
  const resize = () => { const w=stage.clientWidth, h=stage.clientHeight; renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix(); render(); };
  new ResizeObserver(resize).observe(stage); resize();
  renderer.domElement.addEventListener('webglcontextlost', e => {e.preventDefault(); $('status').textContent='Se perdió el contexto gráfico. Recarga la página para recuperar el visor.';});
} catch (error) {
  $('status').textContent = 'WebGL no está disponible. Puedes consultar las imágenes y descargar los modelos.';
  console.error(error);
}
function render(){ if(renderer) renderer.render(scene,camera); }
function dispose(){if(object){scene.remove(object);object.geometry.dispose();object.material.dispose();object=null;}}
function center(){
  if(!object || !controls)return;
  object.geometry.computeBoundingSphere();
  const r=object.geometry.boundingSphere.radius;
  const limitingFov=Math.min(camera.fov*Math.PI/360, Math.atan(Math.tan(camera.fov*Math.PI/360)*camera.aspect));
  const distance=r/Math.sin(limitingFov)*1.15;
  camera.position.copy(new THREE.Vector3(1,.65,1).normalize().multiplyScalar(distance));
  camera.near=Math.max(r/1000,.01);camera.far=r*100;camera.updateProjectionMatrix();
  controls.target.set(0,0,0);controls.minDistance=r*.1;controls.maxDistance=r*20;controls.update();render();
}
function material(geometry){
  if($('mode').value==='points') return new THREE.PointsMaterial({color:0x176e69,size:.45});
  return new THREE.MeshStandardMaterial({color:geometry.hasAttribute('color') ? 0xffffff : 0x299b8c,vertexColors:geometry.hasAttribute('color'),side:THREE.DoubleSide,roughness:.75,wireframe:$('mode').value==='wire'});
}
function setMode(){
  if(!object)return;
  const geometry=object.geometry;scene.remove(object);object.material.dispose();
  object=$('mode').value==='points'?new THREE.Points(geometry,material(geometry)):new THREE.Mesh(geometry,material(geometry));scene.add(object);render();
}
async function json(path){const response=await fetch(path);if(!response.ok)throw new Error(`${path}: HTTP ${response.status}`);return response.json();}
function dimensions(){
  $('dimensions').replaceChildren();
  for(const row of comparison.rows.filter(r=>r.campaign===selected.public_name && r.variant===$('variant').value)){
    const tr=document.createElement('tr');
    for(const value of [labels[row.dimension] || row.dimension,fmt(row.reference_mm),fmt(row.reconstructed_mm),fmt(row.difference_mm)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}
    $('dimensions').append(tr);
  }
}
async function load(){
  const token=++request;abort?.abort();abort=new AbortController();dispose();render();
  const dir=`./resultados/${selected.public_name}/`, filename=filenames[$('variant').value];
  $('download').href=dir+filename;$('download').download=selected.public_name+'_'+filename;
  $('mesh-info').textContent='';dimensions();
  if(!renderer)return;
  $('status').textContent='Descargando modelo…';
  try{
    const response=await fetch(dir+filename,{signal:abort.signal});if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const buffer=await response.arrayBuffer();if(token!==request)return;
    $('status').textContent='Preparando geometría…';
    const geometry=new PLYLoader().parse(buffer);
    if(!geometry.getAttribute('position')?.count){geometry.dispose();throw new Error('Malla vacía');}
    geometry.center(); // Transformaciones solo de presentación: originales intactos.
    geometry.rotateX(Math.PI); // Los resultados usan Y hacia abajo; mostrar la punta hacia arriba.
    geometry.computeVertexNormals();
    object=new THREE.Mesh(geometry,material(geometry));scene.add(object);setMode();center();
    $('mesh-info').textContent=`${fmt(geometry.getAttribute('position').count)} vértices · ${fmt((geometry.index?.count ?? 0)/3)} triángulos · ${(buffer.byteLength/1048576).toFixed(1)} MB · Escala original en mm`;
    $('status').textContent='Modelo listo para explorar';
  }catch(error){if(token!==request || error.name==='AbortError')return;$('status').textContent='No se pudo cargar el modelo. Selecciona otra campaña o vuelve a intentarlo; también puedes descargar el PLY.';console.error(error);}
}
async function select(item){
  selected=item;const name=item.public_name;history.replaceState(null,'',`#${name}`);
  document.querySelectorAll('#campaigns button').forEach(b=>b.setAttribute('aria-current',String(b.dataset.name===name)));
  $('title').textContent=title(name);$('quality').textContent=({accepted:'Aceptado',warning:'Con advertencias',rejected:'Rechazado'})[item.quality] || item.quality;
  $('warnings').textContent=[...(item.warnings||[]),...(item.reject_reasons||[])].join(' ') || 'Sin advertencias en los controles internos. Esto no demuestra exactitud dimensional respecto al objeto físico.';
  const dir=`./resultados/${name}/`;
  $('left').src=dir+'captura_izquierda.png';$('right').src=dir+'captura_derecha.png';$('preview').src=dir+'preview_validacion.png';
  $('report').href=dir+'validacion_17.json';$('source').href=`https://github.com/Mategut/Sistema_3D/tree/main/resultados/${name}`;
  $('metrics').textContent='Cargando informe…';
  load();
  try{
    const v=await json(dir+'validacion_17.json');if(selected!==item)return;
    const g=v.geometry_fidelity;$('metrics').replaceChildren();
    for(const [label,value] of [['Media nube → malla (mm)',g?.cloud_to_mesh_mm?.mean],['P95 nube → malla (mm)',g?.cloud_to_mesh_mm?.p95],['P95 malla → nube (mm)',g?.mesh_to_cloud_mm?.p95]]){
      const box=document.createElement('div');box.className='metric';const strong=document.createElement('strong');strong.textContent=fmt(value);const span=document.createElement('span');span.textContent=label;box.append(strong,span);$('metrics').append(box);
    }
  }catch(error){if(selected===item)$('metrics').textContent='No se pudo cargar el informe de métricas.';}
}
$('variant').addEventListener('change',load);$('mode').addEventListener('change',setMode);$('reset').addEventListener('click',center);
let spinning=false,lastTime=0;
function animate(time){if(!spinning)return;const dt=Math.min((time-lastTime)/1000,.1);lastTime=time;if(controls){controls.update(dt);render();}requestAnimationFrame(animate);}
$('rotate').addEventListener('click',()=>{if(!controls)return;spinning=!spinning;controls.autoRotate=spinning;$('rotate').setAttribute('aria-pressed',String(spinning));if(spinning){lastTime=performance.now();requestAnimationFrame(animate);}});
stage.addEventListener('keydown',e=>{
  if(!controls || !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','0'].includes(e.key))return;e.preventDefault();
  if(e.key==='0'){center();return;}
  const offset=camera.position.clone().sub(controls.target);const s=new THREE.Spherical().setFromVector3(offset);
  if(e.key==='ArrowLeft')s.theta-=.12;if(e.key==='ArrowRight')s.theta+=.12;if(e.key==='ArrowUp')s.phi-=.12;if(e.key==='ArrowDown')s.phi+=.12;
  if(['+','='].includes(e.key))s.radius*=.9;if(e.key==='-')s.radius*=1.1;s.makeSafe();s.radius=THREE.MathUtils.clamp(s.radius,controls.minDistance,controls.maxDistance);
  camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(s));controls.update();render();
});
const [catalog,comparison]=await Promise.all([json('./resultados/catalogo.json'),json('./resultados/comparacion_dimensional.json')]);
for(const item of catalog){const button=document.createElement('button');button.textContent=title(item.public_name);button.dataset.name=item.public_name;button.addEventListener('click',()=>select(item));$('campaigns').append(button);}
await select(catalog.find(c=>c.public_name===location.hash.slice(1)) || catalog[0]);
