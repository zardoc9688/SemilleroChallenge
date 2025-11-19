"""
Visualizador 3D de Cortes Histológicos
Aplicación Streamlit mejorada con colorización FalseColor y estrategia de cubos

Autores: Daniel Yaruro, Juan Mantilla
Fecha: Enero 2025
Dataset: Kather Texture 2016 - Colorectal Histology MNIST
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import base64
from pathlib import Path
import time 
import streamlit.components.v1 as components

# Importar colorización
try:
    from colorization import HistologyColorizer
    COLORIZATION_AVAILABLE = True
except ImportError:
    COLORIZATION_AVAILABLE = False
    st.warning(" Módulo colorization.py no encontrado. Colorización deshabilitada.")

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="Visualizador 3D de Cortes Histológicos",
    page_icon="",
    layout="wide"
)

st.title(" Visualización 3D de Cortes Histológicos")
st.markdown("### Stack 3D de imágenes histológicas colorrectales")
st.markdown("**Dataset:** Kather Texture 2016 - Colorectal Histology MNIST")

# ==================== SIDEBAR ====================
st.sidebar.header(" Configuración")

data_path = st.sidebar.text_input(
    "Ruta del dataset:",
    r"C:\Users\ASUS STRIX\Documents\Semillero\Kather_texture_2016_image_tiles_5000\Kather_texture_2016_image_tiles_5000"
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

# ==================== DOWNSAMPLE EN POTENCIAS DE 2 ====================
st.sidebar.markdown("###  Downsample (Potencias de 2)")
st.sidebar.info(" Según recomendación: usar potencias de 2")

downsample_options = {
    "1x (150×150) - Sin reducción": 1,
    "2x (75×75) - Factor 2¹": 2,
    "4x (37×37) - Factor 2²": 4,
    "8x (18×18) - Factor 2³": 8,
    "16x (9×9) - Factor 2⁴": 16
}

downsample_label = st.sidebar.selectbox(
    "Factor de reducción:",
    options=list(downsample_options.keys())
)
downsample_factor = downsample_options[downsample_label]

# ==================== ESTRATEGIA DE CUBOS (CRÍTICO) ====================
st.sidebar.markdown("###  Estrategia de Volumen")
st.sidebar.warning(" **CRÍTICO:** Para >50 cortes, usar cubos pequeños")

volume_strategy = st.sidebar.radio(
    "Modo de visualización:",
    ["Stack Completo (≤50 cortes)", "Cubos Pequeños (>50 cortes)"]
)

if volume_strategy == "Cubos Pequeños (>50 cortes)":
    cube_size = st.sidebar.slider(
        "Tamaño de cubo:",
        20, 100, 50, 10,
        help="Número de cortes por cubo. Recomendado: 50"
    )
    total_slices = st.sidebar.number_input(
        "Total de cortes:",
        50, 500, 200, 50,
        help="Número total de cortes a cargar"
    )
    num_imagenes = total_slices
else:
    num_imagenes = st.sidebar.slider("Número de cortes:", 5, 50, 30, 5)
    cube_size = num_imagenes

# ==================== COLORIZACIÓN FALSECOLOR ====================
if COLORIZATION_AVAILABLE:
    st.sidebar.markdown("###  Colorización (FalseColor-Python)")
    
    enable_colorization = st.sidebar.checkbox(
        "Habilitar colorización",
        value=True,
        help="Aplicar pseudo-colorización según paper FalseColor-Python"
    )
    
    if enable_colorization:
        colormap = st.sidebar.selectbox(
            "Colormap:",
            ["H&E", "Viridis", "Hot", "Turbo", "Jet", "Parula"]
        )
        apply_clahe = st.sidebar.checkbox("CLAHE (contraste)", value=True)
        apply_leveling = st.sidebar.checkbox("Nivelación intensidad", value=True)
        
        if apply_clahe:
            clahe_clip = st.sidebar.slider("Clip CLAHE:", 1.0, 4.0, 2.0, 0.5)
        else:
            clahe_clip = 2.0
    else:
        colormap = None
        apply_clahe = False
        apply_leveling = False
        clahe_clip = 2.0
else:
    enable_colorization = False
    colormap = None
    apply_clahe = False
    apply_leveling = False
    clahe_clip = 2.0

keep_color_for_slices = st.sidebar.checkbox("Mostrar cortes en color", value=True)

# ==================== RENDERIZADO GPU ====================
st.sidebar.markdown("###  Renderizado GPU")
ray_steps = st.sidebar.slider("Pasos ray-march:", 50, 600, 200, 10)
ray_alpha = st.sidebar.slider("Alpha acumulación:", 0.01, 1.0, 0.05, 0.01)
display_size = st.sidebar.slider("Tamaño visualización (px):", 200, 1200, 600, 50)

# ==================== FUNCIÓN DE CARGA ====================
@st.cache_data(show_spinner=False)
def load_stack(path_str: str, tissue: str, n: int, resize_factor: int, 
               keep_color: bool, colorize: bool, colormap_name: str,
               clahe: bool, leveling: bool, clahe_clip_val: float):
    """
    Carga y procesa stack de imágenes
    
    Args:
        path_str: Ruta al dataset
        tissue: Tipo de tejido
        n: Número de cortes
        resize_factor: Factor de downsample (potencia de 2)
        keep_color: Mantener información RGB
        colorize: Aplicar colorización FalseColor
        colormap_name: Nombre del colormap
        clahe: Aplicar CLAHE
        leveling: Aplicar nivelación
        clahe_clip_val: Valor de clip para CLAHE
        
    Returns:
        stack_gray, stack_rgb, error_msg
    """
    path = Path(path_str) / tissue
    if not path.exists():
        return None, None, f" Carpeta no encontrada: {path}"
    
    archivos = sorted(path.glob("*.*"))
    archivos = [p for p in archivos if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}]
    archivos = archivos[:n]
    
    if len(archivos) == 0:
        return None, None, " No se encontraron imágenes"
    
    gray_list = []
    rgb_list = [] if keep_color else None
    
    for archivo in archivos:
        try:
            img = Image.open(archivo).convert("RGB")
            
            # Downsample según potencia de 2
            if resize_factor > 1:
                new_size = max(150 // resize_factor, 1)
                img = img.resize((new_size, new_size), Image.LANCZOS)
            
            arr = np.asarray(img, dtype=np.uint8)
            
            # Conversión a escala de grises (ITU-R BT.709)
            gray = (0.2126 * arr[:, :, 0] + 
                   0.7152 * arr[:, :, 1] + 
                   0.0722 * arr[:, :, 2]).astype(np.uint8)
            
            gray_list.append(gray)
            
            if keep_color:
                rgb_list.append(arr)
        except Exception as e:
            continue
    
    if len(gray_list) == 0:
        return None, None, " No se cargaron imágenes válidas"
    
    stack_gray = np.stack(gray_list, axis=0).astype(np.uint8)
    
    # Aplicar colorización si está habilitada
    if colorize and COLORIZATION_AVAILABLE and colormap_name:
        stack_rgb_colored = []
        for gray_slice in stack_gray:
            colored = HistologyColorizer.apply_false_color(
                gray_slice,
                colormap=colormap_name,
                apply_clahe=clahe,
                apply_leveling=leveling,
                clahe_clip=clahe_clip_val
            )
            stack_rgb_colored.append(colored)
        stack_rgb = np.stack(stack_rgb_colored, axis=0)
    else:
        stack_rgb = np.stack(rgb_list, axis=0) if keep_color and rgb_list else None
    
    return stack_gray, stack_rgb, None

# ==================== RENDERIZADOR GPU (continuación) ====================
def render_rgb_viewer(stack_rgb, height=800, steps=200, alpha=0.05):
    """Renderiza volumen 3D con Three.js"""
    if stack_rgb is None:
        st.warning("No hay datos RGB para renderizar")
        return
    
    z_r, y_r, x_r, c = stack_rgb.shape
    assert c == 3
    
    arr = np.ascontiguousarray(stack_rgb)
    rgba = np.ones((z_r, y_r, x_r, 4), dtype=np.uint8) * 255
    rgba[..., :3] = arr
    rgba_arr = np.ascontiguousarray(rgba)
    b = rgba_arr.tobytes()
    b64 = base64.b64encode(b).decode('ascii')
    
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
            setStatus('Cargando three.js...');
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
                setStatus('WebGL2 no disponible. El visor 3D requiere WebGL2.');
                throw new Error('WebGL2 not available');
            }

            setStatus('Construyendo escena...');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100);
            camera.position.set(0,0,2);
            const controls = new THREE.OrbitControls(camera, renderer.domElement);

            setStatus('Creando Data3DTexture...');
            const Texture3DClass = THREE.Data3DTexture || THREE.DataTexture3D;
            const texture = new Texture3DClass(data, WIDTH, HEIGHT, DEPTH);
            texture.format = THREE.RGBAFormat;
            texture.type = THREE.UnsignedByteType;
            texture.unpackAlignment = 1;
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
                        for(int i=0;i<600;i++){
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
            setStatus('Visor iniciado');
            animate();

            window.addEventListener('resize', ()=>{
                renderer.setSize(window.innerWidth, window.innerHeight);
                camera.aspect = window.innerWidth/window.innerHeight;
                camera.updateProjectionMatrix();
            });

        }catch(err){
            console.error(err);
            setStatus('Error: ' + (err && err.message ? err.message : err));
        }
    })();
    </script>
    </body>
    </html>
    '''
    
    html = (html_template
            .replace('%%X%%', str(x_r))
            .replace('%%Y%%', str(y_r))
            .replace('%%Z%%', str(z_r))
            .replace('%%DATA%%', b64)
            .replace('%%STEPS%%', str(steps))
            .replace('%%ALPHA%%', str(alpha)))
    
    components.html(html, height=height)

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    " Volumen 3D", 
    " Cortes Interactivos", 
    " Animación",
    " Información"
])

# ========== TAB 1: VOLUMEN 3D ==========
with tab1:
    if 'stack_gray' in st.session_state:
        stack_gray = st.session_state['stack_gray']
        stack_rgb = st.session_state.get('stack_rgb', None)
        tipo = st.session_state.get('tipo', '')
        
        st.header(f"Volumen 3D - {tipo}")
        
        # Información sobre estrategia de cubos
        if volume_strategy == "Cubos Pequeños (>50 cortes)":
            total_cortes = stack_gray.shape[0]
            num_cubos = int(np.ceil(total_cortes / cube_size))
            
            st.info(f" **Estrategia de Cubos Pequeños activada**")
            st.write(f"- Total de cortes: {total_cortes}")
            st.write(f"- Tamaño de cubo: {cube_size}")
            st.write(f"- Número de cubos: {num_cubos}")
            
            cube_selector = st.selectbox(
                "Selecciona cubo a visualizar:",
                range(num_cubos),
                format_func=lambda x: f"Cubo {x+1} (cortes {x*cube_size} - {min((x+1)*cube_size-1, total_cortes-1)})"
            )
            
            # Extraer cubo seleccionado
            start_idx = cube_selector * cube_size
            end_idx = min((cube_selector + 1) * cube_size, total_cortes)
            
            stack_rgb_display = stack_rgb[start_idx:end_idx] if stack_rgb is not None else None
            
            st.success(f" Visualizando cubo {cube_selector+1}/{num_cubos}")
        else:
            stack_rgb_display = stack_rgb
        
        col1, col2 = st.columns(2)
        with col1:
            opacidad = st.slider("Opacidad:", 0.01, 0.8, 0.1, 0.01)
        with col2:
            num_superficies = st.slider("Nivel de detalle:", 5, 50, 15)
        
        if stack_rgb_display is not None:
            if st.button(" Abrir Visor 3D (GPU)", type="primary"):
                st.session_state['viewer_requested'] = True
            
            if st.session_state.get('viewer_requested', False):
                with st.spinner("Cargando visor 3D..."):
                    render_rgb_viewer(stack_rgb_display, height=800, steps=ray_steps, alpha=ray_alpha)
        else:
            st.warning("No hay datos RGB. Habilita 'Mostrar cortes en color'")
        
        st.info(f" Dimensiones: {stack_gray.shape[2]}×{stack_gray.shape[1]}×{stack_gray.shape[0]} (X×Y×Z)")
    else:
        st.info(" Configura parámetros y haz click en 'Generar Visualización 3D'")

# ========== TAB 2: CORTES INTERACTIVOS ==========
with tab2:
    if 'stack_gray' in st.session_state:
        stack_gray = st.session_state['stack_gray']
        stack_rgb = st.session_state.get('stack_rgb', None)
        
        st.header("Explorador de Cortes")
        orientacion = st.radio(
            "Plano de corte:",
            ["Axial (XY)", "Coronal (XZ)", "Sagital (YZ)"],
            horizontal=True
        )
        
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
            st.image(img_color, clamp=True, width=display_size)
            intensidad_media = float(img_gray.mean())
            intensidad_min = float(img_gray.min())
            intensidad_max = float(img_gray.max())
        else:
            fig = go.Figure(data=go.Heatmap(
                z=img_gray,
                colorscale='Viridis',
                colorbar=dict(title="Intensidad")
            ))
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
        st.info(" Carga un stack primero")

# ========== TAB 3: ANIMACIÓN ==========
# ========== TAB 3: ANIMACIÓN ==========
with tab3:
    if 'stack_gray' in st.session_state:
        stack_gray = st.session_state['stack_gray']
        stack_rgb = st.session_state.get('stack_rgb', None)
        
        st.header("Animación de Cortes")
        st.info(" Navegación secuencial por todos los cortes")
        
        frame_idx = st.slider(
            "Frame (corte):",
            0,
            stack_gray.shape[0] - 1,
            0,
            key="animation_slider"
        )
        
        if keep_color_for_slices and stack_rgb is not None:
            st.image(
                stack_rgb[frame_idx],
                clamp=True,
                caption=f"Corte {frame_idx + 1} de {stack_gray.shape[0]}",
                width=display_size
            )
        else:
            fig_anim = go.Figure(data=go.Heatmap(
                z=stack_gray[frame_idx],
                colorscale='Viridis'
            ))
            fig_anim.update_layout(
                title=f"Corte {frame_idx + 1} de {stack_gray.shape[0]}",
                height=600
            )
            st.plotly_chart(fig_anim, use_container_width=True)
        
        # Botón de reproducción
        if st.button(" Reproducir secuencia"):
            placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(stack_gray.shape[0]):
                status_text.text(f"Reproduciendo corte {i+1}/{stack_gray.shape[0]}")
                
                if keep_color_for_slices and stack_rgb is not None:
                    placeholder.image(
                        stack_rgb[i], 
                        clamp=True, 
                        width=display_size,
                        caption=f"Corte {i+1}"
                    )
                else:
                    fig_seq = go.Figure(data=go.Heatmap(
                        z=stack_gray[i],
                        colorscale='Viridis'
                    ))
                    fig_seq.update_layout(
                        title=f"Corte {i+1} de {stack_gray.shape[0]}",
                        height=500
                    )
                    placeholder.plotly_chart(fig_seq, use_container_width=True)
                
                progress_bar.progress((i + 1) / stack_gray.shape[0])
                time.sleep(0.08) 
            
            progress_bar.empty()
            status_text.empty()
            st.success(" Secuencia completada")
    else:
        st.info(" Carga un stack primero")

# ========== TAB 4: INFORMACIÓN ==========
with tab4:
    st.header(" Información del Proyecto")
    
    if 'stack_gray' in st.session_state:
        stack_gray = st.session_state['stack_gray']
        stack_rgb = st.session_state.get('stack_rgb', None)
        tipo = st.session_state.get('tipo', '')
        resize = st.session_state.get('resize', 1)
        
        st.subheader("Estadísticas del Stack Cargado")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tipo de Tejido", tipo)
        col2.metric("Número de Cortes", stack_gray.shape[0])
        col3.metric("Resolución", f"{stack_gray.shape[2]}×{stack_gray.shape[1]}")
        col4.metric("Factor Downsample", f"{downsample_factor}x")
        
        # Uso de memoria
        memory_gray = stack_gray.nbytes / (1024**2)  # MB
        memory_rgb = stack_rgb.nbytes / (1024**2) if stack_rgb is not None else 0
        total_memory = memory_gray + memory_rgb
        
        st.subheader("Uso de Memoria")
        col1, col2, col3 = st.columns(3)
        col1.metric("Stack Gray", f"{memory_gray:.2f} MB")
        col2.metric("Stack RGB", f"{memory_rgb:.2f} MB")
        col3.metric("Total", f"{total_memory:.2f} MB")
        
        # Análisis de intensidad
        st.subheader("Análisis de Intensidad")
        mean_intensity = np.mean(stack_gray)
        std_intensity = np.std(stack_gray)
        min_intensity = np.min(stack_gray)
        max_intensity = np.max(stack_gray)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Media", f"{mean_intensity:.1f}")
        col2.metric("Desv. Estándar", f"{std_intensity:.1f}")
        col3.metric("Mínimo", f"{min_intensity}")
        col4.metric("Máximo", f"{max_intensity}")
        
        # Histograma de intensidades
        st.subheader("Histograma de Intensidades")
        hist_data = stack_gray.flatten()
        
        fig_hist = go.Figure(data=[go.Histogram(
            x=hist_data,
            nbinsx=50,
            marker_color='steelblue'
        )])
        fig_hist.update_layout(
            title="Distribución de Intensidades en el Stack",
            xaxis_title="Intensidad (0-255)",
            yaxis_title="Frecuencia",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Información del proyecto
    st.markdown("---")
    st.subheader(" Acerca del Proyecto")
    
    st.markdown("""
    **Autores:** Daniel Yaruro, Juan Mantilla  
    **Institución:** [Tu institución]  
    **Fecha:** Enero 2025
    
    ### Dataset
    - **Nombre:** Colorectal Histology MNIST (Kather Texture 2016)
    - **Fuente:** [Kaggle](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist)
    - **Referencia:** Kather, J. N., et al. (2016)
    
    ### Tecnologías Utilizadas
    - **Framework:** Streamlit
    - **Visualización 3D:** Three.js (WebGL2)
    - **Procesamiento:** NumPy, OpenCV, PIL
    - **Colorización:** Implementación basada en FalseColor-Python
    
    ### Mejoras Implementadas
    
     **Downsample en potencias de 2** (según recomendación del profesor)  
     **Estrategia de cubos pequeños** para volúmenes grandes (>50 cortes)  
     **Colorización FalseColor-Python** según paper de Giacomelli et al. (2020)  
     **Documentación completa** del preprocesamiento  
     **Referencias bibliográficas** recolectadas
    
    ### Referencias Principales
    
    1. **Giacomelli, M. G., et al. (2020).** "FalseColor-Python: A rapid intensity-leveling 
       and digital-staining package for fluorescence-based slide-free digital pathology." 
       *PLOS ONE*. [DOI: 10.1371/journal.pone.0233198](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0233198)
    
    2. **Kather, J. N., et al. (2016).** "Multi-class texture analysis in colorectal 
       cancer histology." *Scientific Reports*.
    
    3. **ITU-R Recommendation BT.709** - Parameter values for HDTV standards (conversión RGB→Gray)
    
    ### Documentación Adicional
    
     Ver archivo `PREPROCESSING.md` para detalles del preprocesamiento  
     Ver archivo `REFERENCIAS.md` para la lista completa de referencias
    """)
    
    st.markdown("---")
    st.subheader(" Configuración Actual")
    
    config_data = {
        "Parámetro": [
            "Ruta del dataset",
            "Tipo de tejido",
            "Número de cortes",
            "Factor downsample",
            "Estrategia de volumen",
            "Colorización",
            "Colormap",
            "CLAHE",
            "Nivelación intensidad",
            "Ray-march steps",
            "Alpha acumulación"
        ],
        "Valor": [
            data_path.split('\\')[-1] if '\\' in data_path else data_path.split('/')[-1],
            tipo_seleccionado,
            num_imagenes,
            f"{downsample_factor}x",
            volume_strategy,
            "Habilitada" if enable_colorization else "Deshabilitada",
            colormap if enable_colorization else "N/A",
            "Sí" if apply_clahe else "No",
            "Sí" if apply_leveling else "No",
            ray_steps,
            f"{ray_alpha:.2f}"
        ]
    }
    
    import pandas as pd
    df_config = pd.DataFrame(config_data)
    st.dataframe(df_config, use_container_width=True, hide_index=True)

# ==================== BOTÓN DE CARGA ====================
st.sidebar.markdown("---")

if st.sidebar.button(" Generar Visualización 3D", type="primary"):
    # Limpiar estado anterior
    st.session_state['viewer_requested'] = False
    
    with st.spinner(f"Cargando {num_imagenes} imágenes de {tipo_seleccionado}..."):
        stack_gray, stack_rgb, err = load_stack(
            data_path,
            tipo_seleccionado,
            num_imagenes,
            downsample_factor,
            keep_color_for_slices,
            enable_colorization,
            colormap if enable_colorization else None,
            apply_clahe,
            apply_leveling,
            clahe_clip
        )
    
    if err:
        st.error(err)
    else:
        st.success(f" Stack cargado: {stack_gray.shape} (Z,Y,X)")
        
        # Guardar en session state
        st.session_state['stack_gray'] = stack_gray
        st.session_state['stack_rgb'] = stack_rgb
        st.session_state['tipo'] = tipo_seleccionado
        st.session_state['resize'] = downsample_factor
        
        # Información adicional
        memory_mb = (stack_gray.nbytes + (stack_rgb.nbytes if stack_rgb is not None else 0)) / (1024**2)
        st.info(f" Uso de memoria: {memory_mb:.2f} MB")
        
        if enable_colorization and COLORIZATION_AVAILABLE:
            st.success(f" Colorización aplicada: {colormap}")
        
        if volume_strategy == "Cubos Pequeños (>50 cortes)":
            num_cubos = int(np.ceil(num_imagenes / cube_size))
            st.info(f" Dividido en {num_cubos} cubos de {cube_size} cortes")
        
        # Marcar visor como disponible
        if stack_rgb is not None:
            st.session_state['viewer_requested'] = True
        
        # Forzar rerun para actualizar tabs
        st.rerun()

# ==================== MENSAJE INICIAL ====================
if 'stack_gray' not in st.session_state:
    st.info(" Selecciona configuración en el sidebar y haz click en 'Generar Visualización 3D'")
    
    st.markdown("""
    ###  Cómo funciona esta aplicación
    
    Esta herramienta permite visualizar **stacks 3D de imágenes histológicas** mediante:
    
    ####  Características Principales
    
    1. **Volumen 3D Interactivo**
       - Renderizado GPU con WebGL2 y Three.js
       - Rotación, zoom y navegación en tiempo real
       - Estrategia de cubos pequeños para volúmenes grandes
    
    2. **Exploración de Cortes**
       - Tres planos de visualización: Axial, Coronal, Sagital
       - Análisis de intensidades por corte
       - Modo color y escala de grises
    
    3. **Animación Secuencial**
       - Reproducción automática de todos los cortes
       - Navegación frame por frame
    
    4. **Colorización Avanzada (FalseColor-Python)**
       - Simulación de tinción H&E
       - Múltiples colormaps perceptuales
       - CLAHE para mejora de contraste
       - Nivelación de intensidad automática
    
    ####  Mejoras Implementadas (según retroalimentación)
    
     **Downsample en potencias de 2:** 1x, 2x, 4x, 8x, 16x  
     **Estrategia de cubos:** Divide volúmenes >50 cortes en cubos manejables  
     **Colorización científica:** Basada en paper FalseColor-Python (PLOS ONE 2020)  
     **Documentación completa:** Ver `PREPROCESSING.md` y `REFERENCIAS.md`
    
    ####  Pasos para usar
    
    1. **Configura la ruta** del dataset en el sidebar
    2. **Selecciona tipo de tejido** (TUMOR, STROMA, etc.)
    3. **Ajusta downsample** según tu GPU (recomendado: 1x o 2x para máxima calidad)
    4. **Elige estrategia:**
       - Stack completo: ≤50 cortes
       - Cubos pequeños: >50 cortes (más estable)
    5. **Opcional:** Habilita colorización FalseColor
    6. Click en **"Generar Visualización 3D"**
    
    ####  Estructura esperada del dataset
    
    ```
    data/
    └── Kather_texture_2016_image_tiles_5000/
        ├── 01_TUMOR/
        │   ├── imagen001.tif
        │   ├── imagen002.tif
        │   └── ...
        ├── 02_STROMA/
        ├── 03_COMPLEX/
        ├── 04_LYMPHO/
        ├── 05_DEBRIS/
        ├── 06_MUCOSA/
        ├── 07_ADIPOSE/
        └── 08_EMPTY/
    ```
    
    ####  Requisitos Técnicos
    
    - **Navegador:** Chrome, Edge, o Firefox (con WebGL2)
    - **RAM:** Mínimo 4GB (recomendado 8GB)
    - **GPU:** Recomendada para volúmenes >30 cortes
    - **Python:** 3.8 o superior
    
    ####  Referencias
    
    - **Dataset:** [Kaggle - Colorectal Histology MNIST](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist)
    - **Paper FalseColor:** [Giacomelli et al., PLOS ONE 2020](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0233198)
    - **Documentación:** Ver archivos `.md` en el repositorio
    
    ---
    
    **Autores:** Daniel Yaruro, Juan Mantilla  
    **Fecha:** Enero 2025  
    **Versión:** 2.0 (con mejoras según retroalimentación del profesor)
    """)

# ==================== FOOTER ====================
st.sidebar.markdown("---")
st.sidebar.markdown("###  Documentación")
st.sidebar.markdown("""
- `REFERENCIAS.md` - Referencias bibliográficas
""")

st.sidebar.markdown("---")
st.sidebar.info("""
 **Tip:** Para volúmenes grandes (>100 cortes), usa:
- Downsample 2x o 4x
- Estrategia de cubos pequeños
- Tamaño de cubo: 50 cortes
""")