import streamlit as st
from agent import Agent
import folium 
import pandas as pd

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, MeasureControl
import numpy as np
import json

agent = Agent("")

st.set_page_config(layout='wide')
st.title('Tourism intellAgent App')

def initialize_session_state():
    if "center" not in st.session_state:
        st.session_state.center = [48.9, 2.4]
    if "zoom" not in st.session_state:
        st.session_state.zoom = 10
    if "marker" not in st.session_state:
        st.session_state.marker = []
        
def initialize_map(center, zoom):
    if "map" not in st.session_state or st.session_state.map is None:
        st.session_state.center = center
        st.session_state.zoom = zoom
        folium_map = folium.Map(location=st.session_state.center,
                                zoom_start=st.session_state.zoom)
        st.session_state.map = folium_map
    return st.session_state.map

def reset_session_state():
    for key in st.session_state.keys():
        if key in ["center", "zoom"]:
            continue
        del st.session_state[key]
    initialize_session_state()  

st.write("""
    This app predicts the **Tourism**!         
""")

initialize_session_state()
col1, col2 = st.columns(2)

with col1:
    request = st.text_area("Onde você gostaria de ir?")
    button = st.button("Pedir sugestão de roteiro")
    box = st.container(height=300)
    with box:
        container = st.empty()
        container.header("Itinerário")
if button and request:
    reset_session_state()
    itinerary = agent.get_itinerary(request)
    try:
        container.write(itinerary['agent_suggestion'])
    except KeyError:
        container.write("Desculpe, não consegui encontrar um roteiro para você.")
        
    try:
        points_coordinates = []
        days = json.loads(itinerary['coordinates'])
        for day in days.values():
            for item in day:
                locations = item["locations"]
                for loc in locations:
                    points_coordinates.append((loc['lat'], loc['lon']))
        st.session_state["marker"] = [folium.Marker(location=point)
                                      for point in points_coordinates]
    except KeyError:
        pass

    center_info = json.loads(itinerary['center_info'])
    center = center_info['center']
    zoom = center_info['zoom']
    st.session_state.center = center
    st.session_state.zoom = zoom

with col2:
    folium_map = initialize_map(st.session_state.center, st.session_state.zoom)
    fg = folium.FeatureGroup(name="Markers")
    for marker in st.session_state["marker"]:
        fg.add_child(marker)
    fg.add_to(folium_map)
    folium_static(folium_map)