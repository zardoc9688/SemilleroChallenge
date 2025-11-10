import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import base64
import io
from pathlib import Path

# Configuración básica de la página
st.set_page_config(page_title="Visualizador 3D de Cortes Histológicos", layout="wide")

st.title("Visualización 3D de Cortes Histológicos")
st.markdown("### Stack 3D de imágenes histológicas colorrectales")


# Sidebar: opciones de carga y rendimiento
st.sidebar.header("Configuración")

data_path = st.sidebar.text_input(
    "Ruta del dataset:",
    r"C:\Users\danie\3D Objects\Proyecto Histologia\data\Kather_texture_2016_image_tiles_5000"
)

tipo_tejido_opciones = [
    "01_TUMOR",
    "02_STROMA",
    "03_COMPLEX",
    "04_LYMPHO",
    "05_DEBRIS",
    "06_MUCOSA",
    "07_ADIPOSE",
    "08_EMPTY",
]

tipo_seleccionado = st.sidebar.selectbox("Tipo de tejido:", tipo_tejido_opciones)

num_imagenes = st.sidebar.slider("Número de cortes (imágenes):", 5, 500, 30, step=5)

# Opciones para manejar memoria y color
# Nota: las imágenes de origen tienen como máximo 150×150 px en este dataset.
resize_dim = st.sidebar.slider("Downsample (píxeles por lado, 0 = sin cambio, max 150):", 0, 150, 150, step=1)
keep_color_for_slices = st.sidebar.checkbox("Mostrar cortes en color (slices)", value=True)
# Controles para el renderer GPU
ray_steps = st.sidebar.slider("Pasos de ray-march (más = smoother, más lento)", 50, 600, 200, step=10)
ray_alpha = st.sidebar.slider("Alpha de acumulación (0.01-1.0)", 0.01, 1.0, 0.05, step=0.01)
# Tamaño de visualización para cortes (en píxeles)
display_size = st.sidebar.slider("Tamaño visualización de cortes (px)", 200, 1200, 600, step=50)
# Eliminada la opción de volumen en escala de grises: la app renderiza RGB por defecto en el visor GPU


@st.cache_data(show_spinner=False)
def load_stack(path_str: str, tissue: str, n: int, resize: int, keep_color: bool):
    """Carga hasta n imágenes de la carpeta, devuelve stack en gris (Z,Y,X,uint8)
    y opcionalmente stack RGB (Z,Y,X,3,uint8). Se hace resize si resize>0.
    """
    path = Path(path_str) / tissue
    if not path.exists():
        return None, None, f"Carpeta no encontrada: {path}"

    archivos = sorted(path.glob("*.*"))  # aceptar varios formatos
    # filtrar por extensiones de imagen comunes
    archivos = [p for p in archivos if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}]
    archivos = archivos[:n]

    if len(archivos) == 0:
        return None, None, "No se encontraron imágenes en la carpeta especificada."

    gray_list = []
    rgb_list = [] if keep_color else None

    for archivo in archivos:
        try:
            img = Image.open(archivo).convert("RGB")
            # El dataset tiene una resolución máxima nativa de 150x150.
            # Si el usuario solicita un resize mayor, lo limitamos a 150 para evitar
            # generar resoluciones inexistentes y desperdiciar memoria.
            if resize and resize > 0:
                target_r = min(int(resize), 150)
                if target_r != resize:
                    # opcional: podríamos informar al usuario; por ahora simplemente capear
                    pass
                img = img.resize((target_r, target_r), Image.LANCZOS)
            arr = np.asarray(img, dtype=np.uint8)

            # Calcular luminancia (más eficiente y en uint8)
            # Y = 0.2126 R + 0.7152 G + 0.0722 B
            gray = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]).astype(np.uint8)
            gray_list.append(gray)

            if keep_color:
                rgb_list.append(arr)

        except Exception as e:
            # saltar archivos que no se puedan leer
            continue

    if len(gray_list) == 0:
        return None, None, "No se pudieron cargar imágenes válidas (tal vez formato o tamaño incompatible)."

    stack_gray = np.stack(gray_list, axis=0).astype(np.uint8)  # Z,Y,X
    stack_rgb = np.stack(rgb_list, axis=0).astype(np.uint8) if keep_color and rgb_list else None

    return stack_gray, stack_rgb, None

import streamlit.components.v1 as components


def render_rgb_viewer(stack_rgb, height=800, steps=200, alpha=0.05):
        """Genera y muestra el HTML/Three.js que renderiza el volumen RGB usando DataTexture3D.
        stack_rgb: ndarray Z,Y,X,3 uint8
        """
        if stack_rgb is None:
                st.warning("No hay datos RGB para renderizar.")
                return

        z_r, y_r, x_r, c = stack_rgb.shape
        assert c == 3
        arr = np.ascontiguousarray(stack_rgb)
        # three.js recent versions expect RGBA (4 components) for 3D textures; convert RGB->RGBA with alpha=255
        rgba = np.ones((z_r, y_r, x_r, 4), dtype=np.uint8) * 255
        rgba[..., :3] = arr
        rgba_arr = np.ascontiguousarray(rgba)
        b = rgba_arr.tobytes()
        b64 = base64.b64encode(b).decode('ascii')

        # HTML template with placeholders to avoid f-string brace collisions
        html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>html,body{{margin:0;height:100%;background:#000;color:#fff}}#container{{width:100%;height:100%}}#status{{position:fixed;left:10px;top:10px;z-index:9999;padding:8px 12px;background:rgba(0,0,0,0.6);border-radius:6px}}</style>
    </head>
    <body>
    <div id="status">Cargando visor...</div>
    <div id="container"></div>
    <script>
    function setStatus(s){
        const el = document.getElementById('status');
        if(el) el.textContent = s;
        console.log('[VOL_VIEWER]', s);
    }
    function b64ToUint8Array(b64){
        const binary = atob(b64);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
        return bytes;
    }

    async function loadScript(url){
        return new Promise((resolve, reject)=>{
            const s = document.createElement('script');
            s.src = url;
            s.onload = () => resolve();
            s.onerror = (e) => reject(new Error('Failed to load ' + url));
            document.head.appendChild(s);
        });
    }

    setStatus('Decodificando datos...');
    const WIDTH = %%X%%;
    const HEIGHT = %%Y%%;
    const DEPTH = %%Z%%;
    const data = b64ToUint8Array('%%DATA%%');
    setStatus(`Decoded: ${data.length} bytes (expect ${WIDTH*HEIGHT*DEPTH*4} bytes)`);

    (async ()=>{
        try{
            setStatus('Cargando librerías (three.js)...');
            // cargar three.js y OrbitControls dinámicamente y en orden
            await loadScript('https://unpkg.com/three@0.146.0/build/three.min.js');
            setStatus('three.js cargado');
            await loadScript('https://unpkg.com/three@0.146.0/examples/js/controls/OrbitControls.js');
            setStatus('OrbitControls cargado');

            setStatus('Creando renderer WebGL...');
            const container = document.getElementById('container');
            const renderer = new THREE.WebGLRenderer({antialias:true});
            renderer.setSize(window.innerWidth, window.innerHeight);
            container.appendChild(renderer.domElement);
            setStatus('Renderer creado');

            const gl = renderer.getContext();
            const isWebGL2 = (typeof WebGL2RenderingContext !== 'undefined') && (gl instanceof WebGL2RenderingContext);
            setStatus('WebGL2 available: ' + isWebGL2);

            if(!isWebGL2){
                setStatus('WebGL2 no disponible en este contexto. El visor 3D requiere WebGL2 (DataTexture3D). Abre DevTools para más info.');
                throw new Error('WebGL2 not available');
            }

            setStatus('Construyendo escena...');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100);
            camera.position.set(0,0,2);
            const controls = new THREE.OrbitControls(camera, renderer.domElement);

            // Crear Data3DTexture (API renombrada en three.js recientes)
            setStatus('Creando Data3DTexture...');
            const Texture3DClass = THREE.Data3DTexture || THREE.DataTexture3D;
            const texture = new Texture3DClass(data, WIDTH, HEIGHT, DEPTH);
            // recent three.js removed RGBFormat; use RGBA and provide 4 components per voxel
            texture.format = THREE.RGBAFormat;
            texture.type = THREE.UnsignedByteType;
            texture.unpackAlignment = 1;
            // use linear filtering to smooth slice boundaries (trilinear sampling)
            texture.magFilter = THREE.LinearFilter;
            texture.minFilter = THREE.LinearFilter;
            texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
            texture.generateMipmaps = false;
            texture.needsUpdate = true;

            const geometry = new THREE.BoxGeometry(1, HEIGHT/WIDTH, DEPTH/WIDTH);
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    u_data: { value: texture },
                    u_size: { value: new THREE.Vector3(WIDTH, HEIGHT, DEPTH) },
                    u_steps: { value: %%STEPS%% },
                    u_alpha: { value: %%ALPHA%% }
                },
                vertexShader: `
                    varying vec3 v_position;
                    void main(){
                        v_position = position;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
                    }
                `,
                fragmentShader: `
                    precision highp float;
                    precision highp sampler3D;
                    varying vec3 v_position;
                    uniform sampler3D u_data;
                    uniform vec3 u_size;
                    uniform int u_steps;
                    uniform float u_alpha;

                    vec2 intersectBox(vec3 orig, vec3 dir){
                        vec3 boxMin = vec3(-0.5);
                        vec3 boxMax = vec3(0.5);
                        vec3 invDir = 1.0 / dir;
                        vec3 tmin_tmp = (boxMin - orig) * invDir;
                        vec3 tmax_tmp = (boxMax - orig) * invDir;
                        vec3 tmin = min(tmin_tmp, tmax_tmp);
                        vec3 tmax = max(tmin_tmp, tmax_tmp);
                        float t0 = max(max(tmin.x, tmin.y), tmin.z);
                        float t1 = min(min(tmax.x, tmax.y), tmax.z);
                        return vec2(t0, t1);
                    }

                    void main(){
                        vec3 rayDir = normalize(v_position - vec3(0.0,0.0,2.0));
                        vec3 rayOrig = v_position;
                        vec2 bounds = intersectBox(rayOrig, rayDir);
                        if(bounds.x > bounds.y) discard;
                        float t = max(bounds.x, 0.0);
                        float tEnd = bounds.y;
                        float dt = (tEnd - t) / float(u_steps);
                        vec3 accColor = vec3(0.0);
                        float accAlpha = 0.0;
                        for(int i=0;i<200;i++){
                            if(i >= u_steps) break;
                            vec3 p = rayOrig + (t + dt*float(i)) * rayDir;
                            vec3 texCoord = p + vec3(0.5);
                            vec3 col = texture(u_data, texCoord).rgb;
                            float lumin = (col.r + col.g + col.b) / 3.0;
                            float alpha = clamp(lumin * u_alpha * 10.0, 0.0, 1.0);
                            accColor = accColor + (1.0 - accAlpha) * col * alpha;
                            accAlpha = accAlpha + (1.0 - accAlpha) * alpha;
                            if(accAlpha >= 0.95) break;
                        }
                        gl_FragColor = vec4(accColor, accAlpha);
                    }
                `,
                transparent: true,
                depthWrite: false
            });

            const mesh = new THREE.Mesh(geometry, material);
            scene.add(mesh);

            function animate(){
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            setStatus('Iniciando animación...');
            animate();

            window.addEventListener('resize', ()=>{
                renderer.setSize(window.innerWidth, window.innerHeight);
                camera.aspect = window.innerWidth/window.innerHeight;
                camera.updateProjectionMatrix();
            });

        }catch(err){
            console.error(err);
            setStatus('Error al inicializar visor: ' + (err && err.message ? err.message : err));
        }
    })();

    </script>
    </body>
    </html>
        '''

        html = html_template.replace('%%X%%', str(x_r)).replace('%%Y%%', str(y_r)).replace('%%Z%%', str(z_r)).replace('%%DATA%%', b64).replace('%%STEPS%%', str(steps)).replace('%%ALPHA%%', str(alpha))
        components.html(html, height=height)


# Place the main tabs here (after helper definitions) so they render at top but have access to functions/vars
tab1, tab2, tab3 = st.tabs(["Volumen 3D", "Cortes Interactivos", "Animación"])

with tab1:
    if 'stack_gray' in st.session_state:
        stack_gray = st.session_state['stack_gray']
        stack_rgb = st.session_state.get('stack_rgb', None)
        tipo = st.session_state.get('tipo', '')

        st.header(f"Volumen 3D - {tipo}")
        col1, col2 = st.columns(2)
        with col1:
            opacidad = st.slider("Opacidad:", 0.01, 0.8, 0.1, 0.01)
        with col2:
            num_superficies = st.slider("Nivel de detalle:", 5, 50, 15)

        if stack_rgb is not None:
            st.success("Visor RGB disponible: el visor GPU está listo. Haz click para abrirlo en esta pestaña.")
            if st.button("Abrir visor RGB (GPU)"):
                st.session_state['viewer_requested'] = True
            # Si el usuario solicitó el visor, renderízalo dentro de esta pestaña
            if st.session_state.get('viewer_requested', False):
                render_rgb_viewer(stack_rgb, height=800, steps=ray_steps, alpha=ray_alpha)
        else:
            st.info("No hay datos RGB disponibles para el volumen. Asegúrate de habilitar 'Mostrar cortes en color' o que las imágenes sean RGB.")

        st.info(f"Dimensiones del volumen (X×Y×Z): {stack_gray.shape[2]}×{stack_gray.shape[1]}×{stack_gray.shape[0]}")
    else:
        st.info("Aún no hay un stack cargado. Selecciona el tipo de tejido y haz click en 'Generar Visualización 3D' en la barra lateral.")

with tab2:
    if 'stack_gray' in st.session_state:
        stack_gray = st.session_state['stack_gray']
        stack_rgb = st.session_state.get('stack_rgb', None)

        st.header("Explorador de Cortes")
        orientacion = st.radio("Plano de corte:", ["Axial (XY)", "Coronal (XZ)", "Sagital (YZ)"], horizontal=True)

        if orientacion == "Axial (XY)":
            max_idx = stack_gray.shape[0] - 1
            slice_idx = st.slider("Índice de corte:", 0, max_idx, max_idx // 2)
            img_gray = stack_gray[slice_idx, :, :]
            img_color = stack_rgb[slice_idx] if stack_rgb is not None else None
            titulo = f"Corte Axial - Profundidad: {slice_idx}/{max_idx}"

        elif orientacion == "Coronal (XZ)":
            max_idx = stack_gray.shape[1] - 1
            slice_idx = st.slider("Índice de corte:", 0, max_idx, max_idx // 2)
            img_gray = stack_gray[:, slice_idx, :]
            img_color = stack_rgb[:, slice_idx, :] if stack_rgb is not None else None
            titulo = f"Corte Coronal - Y: {slice_idx}/{max_idx}"

        else:  # Sagital
            max_idx = stack_gray.shape[2] - 1
            slice_idx = st.slider("Índice de corte:", 0, max_idx, max_idx // 2)
            img_gray = stack_gray[:, :, slice_idx]
            img_color = stack_rgb[:, :, slice_idx] if stack_rgb is not None else None
            titulo = f"Corte Sagital - X: {slice_idx}/{max_idx}"

        st.subheader(titulo)

        if keep_color_for_slices and img_color is not None:
            # Mostrar la imagen en color escalada al tamaño seleccionado para facilitar la visualización
            st.image(img_color, clamp=True, width=display_size)
            intensidad_media = float(img_gray.mean())
            intensidad_min = float(img_gray.min())
            intensidad_max = float(img_gray.max())
        else:
            fig = go.Figure(data=go.Heatmap(z=img_gray, colorscale='Viridis', colorbar=dict(title="Intensidad")))
            fig.update_layout(title=titulo, height=600, xaxis_title='X', yaxis_title='Y')
            st.plotly_chart(fig, use_container_width=True)
            intensidad_media = float(img_gray.mean())
            intensidad_min = float(img_gray.min())
            intensidad_max = float(img_gray.max())

        col1, col2, col3 = st.columns(3)
        col1.metric("Intensidad Media", f"{intensidad_media:.1f}")
        col2.metric("Intensidad Mín", f"{intensidad_min:.1f}")
        col3.metric("Intensidad Máx", f"{intensidad_max:.1f}")
    else:
        st.info("Aún no hay un stack cargado. Genera la visualización 3D desde la barra lateral para explorar cortes.")

with tab3:
    if 'stack_gray' in st.session_state:
        stack_gray = st.session_state['stack_gray']
        stack_rgb = st.session_state.get('stack_rgb', None)

        st.header("Animación de Cortes")
        st.info("Navegación secuencial por todos los cortes histológicos")
        frame_idx = st.slider("Frame (corte):", 0, stack_gray.shape[0] - 1, 0, key="animation_slider")

        if keep_color_for_slices and stack_rgb is not None:
            st.image(stack_rgb[frame_idx], clamp=True, caption=f"Corte {frame_idx + 1} de {stack_gray.shape[0]}", width=display_size)
        else:
            fig_anim = go.Figure(data=go.Heatmap(z=stack_gray[frame_idx], colorscale='Viridis'))
            fig_anim.update_layout(title=f"Corte {frame_idx + 1} de {stack_gray.shape[0]}", height=600)
            st.plotly_chart(fig_anim, use_container_width=True)

        if st.button("Reproducir secuencia"):
            placeholder = st.empty()
            for i in range(stack_gray.shape[0]):
                if keep_color_for_slices and stack_rgb is not None:
                    placeholder.image(stack_rgb[i], clamp=True, width=display_size)
                else:
                    fig_seq = go.Figure(data=go.Heatmap(z=stack_gray[i], colorscale='Viridis'))
                    fig_seq.update_layout(height=500)
                    placeholder.plotly_chart(fig_seq, use_container_width=True)
                st.sleep(0.08)
    else:
        st.info("Aún no hay un stack cargado. Genera la visualización 3D desde la barra lateral para ver la animación.")

if st.sidebar.button("Generar Visualización 3D"):
    st.info(f"Cargando hasta {num_imagenes} imágenes de {tipo_seleccionado}...")
    stack_gray, stack_rgb, err = load_stack(data_path, tipo_seleccionado, num_imagenes, resize_dim, keep_color_for_slices)

    if err:
        st.error(err)
    else:
        st.success(f"Stack cargado: {stack_gray.shape} (Z,Y,X) dtype={stack_gray.dtype}")
        st.session_state['stack_gray'] = stack_gray
        st.session_state['stack_rgb'] = stack_rgb
        st.session_state['tipo'] = tipo_seleccionado
        st.session_state['resize'] = resize_dim
        # Registrar stack en session_state y marcar que el visor debe mostrarse en la pestaña "Volumen 3D"
        if stack_rgb is not None:
            st.session_state['viewer_requested'] = True
        else:
            st.warning("No se cargaron datos RGB. Asegúrate de habilitar 'Mostrar cortes en color' o que las imágenes sean RGB.")


if 'stack_gray' not in st.session_state:
    st.info("Selecciona el tipo de tejido y haz click en 'Generar Visualización 3D' en la barra lateral.")
    st.markdown(
        """
    ### Cómo funciona:

    Esta aplicación crea un **stack 3D** apilando múltiples imágenes histológicas del mismo tipo de tejido.

    **Características:**
    - **Volumen 3D**: Visualización volumétrica interactiva
    - **Cortes**: Explora el volumen en diferentes planos (axial, coronal, sagital)
    - **Animación**: Navega secuencialmente por todos los cortes

    **Requisitos:**
    1. Dataset descargado y descomprimido en la ruta seleccionada
    2. Selecciona un tipo de tejido
    3. Define cuántos cortes quieres apilar y el downsample si lo necesitas
    4. Click en "Generar Visualización 3D"

    ### Estructura esperada:
    ```
    data/
    └── Kather_texture_2016_image_tiles_5000/
        ├── 01_TUMOR/
        ├── 02_STROMA/
        ├── 03_COMPLEX/
        └── ...
    ```
    """
    )

