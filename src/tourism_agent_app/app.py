import streamlit as st
from agent import Agent


agent = Agent("")

st.set_page_config('wide')
st.title('Tourism intellAgent App')

st.write("""
    This app predicts the **Tourism**!         
""")

col1, col2 = st.columns(2)

with col1:
    request = st.text_area("Onde você gostaria de ir?")
    button = st.button("Pedir sugestão de roteiro")
    box = st.container(height=300)
    with box:
        container = st.empty()
        container.header("Itinerário")
if button and request:
    itinerary = agent.get_itinerary(request)
    try:
        container.write(itinerary['agent_suggestion'] + 
                        "<br> Coordenadas:" + 
                        itinerary['coordinates'])

    except KeyError:
        container.write("Desculpe, não consegui encontrar um roteiro para você.")