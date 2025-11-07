import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd

# Configuración
st.set_page_config(
    page_title="Visualizador 3D de Cortes Histológicos",

    layout="wide"
)

st.title(" Visualización 3D de Cortes Histológicos")
st.markdown("### Stack 3D de imágenes histológicas colorrectales")

# Sidebar
st.sidebar.header(" Configuración")

# Ruta del dataset
data_path = st.sidebar.text_input(
    "Ruta del dataset:", 
    r"C:\Users\juanc\Videos\Semillero\data\Kather_texture_2016_image_tiles_5000"
)

# Selector de tipo de tejido
tipo_tejido_opciones = [
    "01_TUMOR",
    "02_STROMA", 
    "03_COMPLEX",
    "04_LYMPHO",
    "05_DEBRIS",
    "06_MUCOSA",
    "07_ADIPOSE",
    "08_EMPTY"
]

tipo_seleccionado = st.sidebar.selectbox(
    "Tipo de tejido:",
    tipo_tejido_opciones
)

# Número de imágenes para el stack
num_imagenes = st.sidebar.slider(
    "Número de cortes (imágenes):",
    min_value=10,
    max_value=100,
    value=30,
    step=5
)

# Botón para cargar
if st.sidebar.button(" Generar Visualización 3D", type="primary"):
    path = Path(data_path) / tipo_seleccionado
    
    if not path.exists():
        st.error(f" Carpeta no encontrada: {path}")
    else:
        st.info(f"Cargando {num_imagenes} cortes histológicos...")
        
        # Cargar imágenes
        imagenes = []
        archivos = list(path.glob("*.tif"))[:num_imagenes]
        
        progress_bar = st.progress(0)
        
        for idx, archivo in enumerate(archivos):
            try:
                img = Image.open(archivo)
                img_array = np.array(img)
                
                if img_array.shape == (150, 150, 3):
                    # Convertir a escala de grises para mejor visualización
                    img_gray = np.mean(img_array, axis=2)
                    imagenes.append(img_gray)
                    
            except Exception as e:
                continue
            
            progress_bar.progress((idx + 1) / len(archivos))
        
        progress_bar.empty()
        
        if len(imagenes) == 0:
            st.error(" No se pudieron cargar imágenes")
        else:
            # Crear stack 3D
            stack_3d = np.array(imagenes)
            st.success(f" Stack 3D creado: {stack_3d.shape}")
            
            # Guardar en session state
            st.session_state['stack_3d'] = stack_3d
            st.session_state['tipo'] = tipo_seleccionado

# Mostrar visualización si hay datos
if 'stack_3d' in st.session_state:
    stack_3d = st.session_state['stack_3d']
    tipo = st.session_state['tipo']
    
    # Crear tabs
    tab1, tab2, tab3 = st.tabs([
        "Volumen 3D",
        "Cortes Interactivos", 
        "Animación"
    ])
    
    # ========== TAB 1: VOLUMEN 3D ==========
    with tab1:
        st.header(f"Volumen 3D - {tipo}")
        
        col1, col2 = st.columns(2)
        with col1:
            opacidad = st.slider("Opacidad:", 0.01, 0.5, 0.1, 0.01)
        with col2:
            num_superficies = st.slider("Nivel de detalle:", 5, 30, 15)
        
        with st.spinner("Generando volumen 3D..."):
            z, y, x = stack_3d.shape
            
            # Crear coordenadas
            X, Y, Z = np.mgrid[0:x, 0:y, 0:z]
            
            # Crear figura con volumen
            fig = go.Figure(data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=stack_3d.T.flatten(),
                isomin=stack_3d.min(),
                isomax=stack_3d.max(),
                opacity=opacidad,
                surface_count=num_superficies,
                colorscale='Viridis',
                caps=dict(x_show=False, y_show=False, z_show=False)
            ))
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='X (píxeles)',
                    yaxis_title='Y (píxeles)',
                    zaxis_title='Profundidad (cortes)',
                    aspectmode='data'
                ),
                height=700,
                title=f"Visualización Volumétrica 3D - {tipo}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.info(f" Dimensiones del volumen: {stack_3d.shape[2]}×{stack_3d.shape[1]}×{stack_3d.shape[0]} píxeles")
    
    # ========== TAB 2: CORTES INTERACTIVOS ==========
    with tab2:
        st.header("Explorador de Cortes")
        
        # Selector de orientación
        orientacion = st.radio(
            "Plano de corte:",
            ["Axial (XY)", "Coronal (XZ)", "Sagital (YZ)"],
            horizontal=True
        )
        
        if orientacion == "Axial (XY)":
            max_idx = stack_3d.shape[0] - 1
            slice_idx = st.slider("Índice de corte:", 0, max_idx, max_idx // 2)
            img_slice = stack_3d[slice_idx, :, :]
            titulo = f"Corte Axial - Profundidad: {slice_idx}/{max_idx}"
            
        elif orientacion == "Coronal (XZ)":
            max_idx = stack_3d.shape[1] - 1
            slice_idx = st.slider("Índice de corte:", 0, max_idx, max_idx // 2)
            img_slice = stack_3d[:, slice_idx, :]
            titulo = f"Corte Coronal - Y: {slice_idx}/{max_idx}"
            
        else:  # Sagital
            max_idx = stack_3d.shape[2] - 1
            slice_idx = st.slider("Índice de corte:", 0, max_idx, max_idx // 2)
            img_slice = stack_3d[:, :, slice_idx]
            titulo = f"Corte Sagital - X: {slice_idx}/{max_idx}"
        
        # Mostrar corte
        fig = go.Figure(data=go.Heatmap(
            z=img_slice,
            colorscale='Viridis',
            colorbar=dict(title="Intensidad")
        ))
        
        fig.update_layout(
            title=titulo,
            height=600,
            xaxis_title='X',
            yaxis_title='Y'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar estadísticas del corte
        col1, col2, col3 = st.columns(3)
        col1.metric("Intensidad Media", f"{np.mean(img_slice):.1f}")
        col2.metric("Intensidad Mín", f"{np.min(img_slice):.1f}")
        col3.metric("Intensidad Máx", f"{np.max(img_slice):.1f}")
    
    # ========== TAB 3: ANIMACIÓN ==========
    with tab3:
        st.header("Animación de Cortes")
        
        st.info("📹 Navegación secuencial por todos los cortes histológicos")
        
        # Slider de animación
        frame_idx = st.slider(
            "Frame (corte):",
            0,
            stack_3d.shape[0] - 1,
            0,
            key="animation_slider"
        )
        
        # Mostrar frame actual
        fig_anim = go.Figure(data=go.Heatmap(
            z=stack_3d[frame_idx, :, :],
            colorscale='Viridis',
            colorbar=dict(title="Intensidad")
        ))
        
        fig_anim.update_layout(
            title=f"Corte {frame_idx + 1} de {stack_3d.shape[0]}",
            height=600,
            xaxis_title='X (píxeles)',
            yaxis_title='Y (píxeles)'
        )
        
        st.plotly_chart(fig_anim, use_container_width=True)
        
        # Botón de reproducción automática
        if st.button(" Reproducir secuencia"):
            placeholder = st.empty()
            
            for i in range(stack_3d.shape[0]):
                fig_seq = go.Figure(data=go.Heatmap(
                    z=stack_3d[i, :, :],
                    colorscale='Viridis'
                ))
                
                fig_seq.update_layout(
                    title=f"Corte {i + 1} de {stack_3d.shape[0]}",
                    height=500
                )
                
                placeholder.plotly_chart(fig_seq, use_container_width=True)
                
                # Pausa entre frames
                import time
                time.sleep(0.1)

else:
    # Mensaje inicial
    st.info("Selecciona el tipo de tejido y haz click en 'Generar Visualización 3D'")
    
    st.markdown("""
    ###  Cómo funciona:
    
    Esta aplicación crea un **stack 3D** apilando múltiples imágenes histológicas del mismo tipo de tejido.
    
    **Características:**
    -  **Volumen 3D**: Visualización volumétrica interactiva
    -  **Cortes**: Explora el volumen en diferentes planos (axial, coronal, sagital)
    -  **Animación**: Navega secuencialmente por todos los cortes
    
    **Requisitos:**
    1. Dataset descargado y descomprimido en `data/`
    2. Selecciona un tipo de tejido
    3. Define cuántos cortes quieres apilar
    4. Click en "Generar Visualización 3D"
    
    ###  Estructura esperada:
    ```
    data/
    └── Kather_texture_2016_image_tiles_5000/
        ├── 01_TUMOR/
        ├── 02_STROMA/
        ├── 03_COMPLEX/
        └── ...
    ```

    """)
