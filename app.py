
import streamlit as st
import h5py
import numpy as np
import os
import glob

# --- Configuration ---
st.set_page_config(page_title="HDF5 Patch Viewer", layout="wide")

# --- Functions ---

@st.cache_data
def get_h5_files(directory):
    """Find all .h5 files in a given directory."""
    return glob.glob(os.path.join(directory, "*.h5"))

@st.cache_data
def get_file_info(file_path):
    """Reads an HDF5 file and returns basic info and dataset keys."""
    try:
        with h5py.File(file_path, 'r') as hf:
            info = {
                "filename": os.path.basename(file_path),
                "num_patches": 0,
                "patch_shape": "N/A",
                "coords_shape": "N/A",
                "patch_dtype": "N/A"
            }
            if 'patches' in hf:
                info["num_patches"] = hf['patches'].shape[0]
                info["patch_shape"] = hf['patches'].shape[1:]
                info["patch_dtype"] = str(hf['patches'].dtype)
            if 'coords' in hf:
                info["coords_shape"] = hf['coords'].shape
            return info
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return None

# --- UI ---

st.title("HDF5 Patch Viewer")
st.markdown("An application to browse and view image patches stored in HDF5 files.")

# --- Sidebar ---
with st.sidebar:
    st.header("File Selection")
    h5_files = get_h5_files('.')
    
    if not h5_files:
        st.warning("No `.h5` files found in the current directory.")
        st.stop()

    selected_file = st.selectbox(
        "Select an HDF5 file:",
        h5_files,
        format_func=os.path.basename
    )

# --- Main Content ---
if selected_file:
    info = get_file_info(selected_file)

    if info:
        st.header(f"File Information: `{info['filename']}`")
        
        # Display file info in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patches", f"{info['num_patches']:,}")
        with col2:
            st.metric("Patch Shape", str(info['patch_shape']))
        with col3:
            st.metric("Patch Data Type", info['patch_dtype'].upper())

        if info["num_patches"] == 0:
            st.warning("This file contains no patches to display.")
            st.stop()

        # --- Viewer Tabs ---
        tab1, tab2 = st.tabs(["Single Patch Viewer", "Grid View"])

        with h5py.File(selected_file, 'r') as hf:
            patches_dset = hf['patches']
            coords_dset = hf.get('coords') # Use .get for safety

            # --- Tab 1: Single Patch Viewer ---
            with tab1:
                st.header("Single Patch Viewer")
                
                patch_index = st.slider(
                    "Select Patch Index:",
                    min_value=0,
                    max_value=info["num_patches"] - 1,
                    value=0,
                    step=1
                )
                
                col_img, col_info = st.columns([3, 1])

                with col_img:
                    st.image(patches_dset[patch_index], use_column_width='always')
                
                with col_info:
                    st.subheader("Patch Details")
                    st.write(f"**Index:** {patch_index}")
                    if coords_dset is not None:
                        coords = coords_dset[patch_index]
                        st.write(f"**Coordinates (X, Y):** `{coords[0]}, {coords[1]}`")
                    else:
                        st.write("**Coordinates:** Not available")

            # --- Tab 2: Grid View ---
            with tab2:
                st.header("Grid View")
                
                cols_per_row = st.select_slider(
                    "Patches per row:",
                    options=[2, 3, 4, 5, 6, 8],
                    value=4
                )
                
                patches_per_page = st.number_input("Patches per page:", min_value=1, max_value=100, value=cols_per_row * 2)

                total_pages = (info["num_patches"] + patches_per_page - 1) // patches_per_page
                
                page_num = st.number_input(
                    "Page Number:",
                    min_value=1,
                    max_value=total_pages,
                    value=1
                )
                
                st.markdown(f"_Showing page **{page_num}** of **{total_pages}**_")

                start_idx = (page_num - 1) * patches_per_page
                end_idx = min(start_idx + patches_per_page, info["num_patches"])

                # Display patches in a grid
                for i in range(start_idx, end_idx, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        patch_idx = i + j
                        if patch_idx < end_idx:
                            with cols[j]:
                                st.image(patches_dset[patch_idx], caption=f"Index: {patch_idx}", use_column_width='always')

else:
    st.info("Please select an HDF5 file from the sidebar to begin.")

