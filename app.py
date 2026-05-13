import streamlit as st
import numpy as np
import torch
from PIL import Image
from pathlib import sys, Path
from src.planner.visualize import colorize_segmentation, draw_path
from src.planner.astar import AStarPlanner
from src.model.unet import MarsUNet

st.set_page_config(page_title="Mars Terrain Classifier", page_icon="🔴", layout="wide")

st.title("🔴 Mars Terrain Classifier")
st.caption("Autonomous terrain perception and path planning from NASA rover imagery")

with st.sidebar:
    st.sidebar.header("Path Planner Costs")
    soil_cost = st.sidebar.slider("soil", 0.1, 20.0, 1.0)
    sand_cost = st.sidebar.slider("sand", 0.1, 20.0, 3.0)
    bedrock_cost = st.sidebar.slider("bedrock", 0.1, 20.0, 2.0)
    large_rock_cost = st.sidebar.slider("large rock", 0.1, 20.0, 10.0)
    rover_track_cost = st.sidebar.slider("rover track", 0.1, 20.0, 0.5)


@st.cache_resource
def load_image():
    path = next(
        Path("data/ai4mars/ai4mars-dataset-merged-0.6/msl/ncam/images/edr").glob(
            "*.JPG"
        )
    )
    image = Image.open(path)
    image = image.resize((512, 512))
    return image


image = load_image()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Image")
    st.image(image, width="stretch")

with col2:
    st.subheader("Terrain Segmentation")
    seg = np.random.randint(0, 5, (512, 512))
    img = colorize_segmentation(seg)
    st.image(img)

with col3:
    st.subheader("Planned Path")
    costs = {
        "rover_track": rover_track_cost,
        "soil": soil_cost,
        "bedrock": bedrock_cost,
        "sand": sand_cost,
        "large_rock": large_rock_cost,
    }
    planner = AStarPlanner(costs)
    path = planner.plan(seg)
    if path is not None:
        path_img = draw_path(img, path)
        st.image(path_img)
    else:
        st.warning("No path found")