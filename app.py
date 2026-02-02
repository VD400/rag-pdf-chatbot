import streamlit as st
import requests

if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"]
else:
    API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG with FastAPI", page_icon="⚡")
st.title("Client-Server RAG ⚡")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if "file_processed" not in st.session_state:
        with st.spinner("Uploading to FastAPI Backend..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                st.success("Backend is ready!")
                st.session_state.file_processed = True
            else:
                st.error(f"Error: {response.text}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask your question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role" : "user", "content" : prompt})

    with st.chat_message("assistant"):
        with st.spinner("Waiting for API response..."):
            try:
                api_response = requests.post(f"{API_URL}/chat", json={"query":prompt})
                if api_response.status_code == 200:
                    answer = api_response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role":"assistant", "content": answer})
                else:
                    st.error(f"API Error: {api_response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Is 'api.py' running?")
            