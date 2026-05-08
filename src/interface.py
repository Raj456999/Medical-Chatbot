import streamlit as st
from rag_pipeline import get_answer
import base64
import os

# FUNCTION TO LOAD IMAGE
def get_base64_image(image_name):

    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, image_name)

    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(
            img_file.read()
        ).decode()

    return encoded

# LOAD IMAGE
bg_image = get_base64_image("bot.jpg")

# PAGE CONFIG
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="centered"
)

# CUSTOM CSS
st.markdown(f"""
<style>

/* Main App Background */
.stApp {{
    background-image: linear-gradient(
        rgba(0,0,0,0.7),
        rgba(0,0,0,0.7)
    ),
    url("data:image/jpg;base64,{bg_image}");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: #111827;
}}

/* Title */
.main-title {{
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: white;
    margin-bottom: 20px;
}}

/* Chat Input */
.stChatInput input {{
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 15px !important;
    border: 2px solid #3b82f6 !important;
}}

/* User Chat Bubble */
.user-message {{
    background-color: #2563eb;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    color: white;
}}

/* Assistant Chat Bubble */
.assistant-message {{
    background-color: #1e293b;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    color: white;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    color: white;
    border-radius: 12px;
    border: none;
    height: 45px;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}}

.stButton > button:hover {{
    transform: scale(1.02);
    background: linear-gradient(90deg, #2563eb, #0891b2);
}}

</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("🩺 Medical AI")
st.sidebar.markdown("---")
st.sidebar.write("AI-powered medical chatbot using RAG architecture.")
st.sidebar.write("Built with:")
st.sidebar.write("• LangChain")
st.sidebar.write("• Pinecone")
st.sidebar.write("• Groq LLM")
st.sidebar.write("• Streamlit")

# TITLE
st.markdown(
    "<div class='main-title'>🩺 Medical AI Assistant</div>",
    unsafe_allow_html=True
)

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# DISPLAY CHAT HISTORY
for message in st.session_state.messages:

    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="user-message">
            👤 {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="assistant-message">
            🤖 {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# CHAT INPUT
query = st.chat_input("Ask your medical question...")

if query:

    # STORE USER MESSAGE
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # DISPLAY USER MESSAGE
    st.markdown(
        f"""
        <div class="user-message">
        👤 {query}
        </div>
        """,
        unsafe_allow_html=True
    )

    # GENERATE RESPONSE
    with st.spinner("Analyzing medical information..."):

        response = get_answer(query)

    # STORE ASSISTANT RESPONSE
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    # DISPLAY RESPONSE
    st.markdown(
        f"""
        <div class="assistant-message">
        🤖 {response}
        </div>
        """,
        unsafe_allow_html=True
    )
