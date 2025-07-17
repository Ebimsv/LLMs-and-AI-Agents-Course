import streamlit as st
from openai import OpenAI
import os
import time
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

@st.cache_data(show_spinner=False)
def ask_llm(prompt, model="gpt-4o-mini-2024-07-18"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Initialize session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠 Intelligent LLM Assistant")
st.markdown("Ask a question, and the agent will try to clarify it before answering.")

# Input + Logic
user_input = st.text_input("Your question:")

if st.button("Ask"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🤔 Agent is clarifying your question..."):
            clarified = ask_llm(f"Can you rephrase this question more clearly: {user_input}")
            time.sleep(0.5)

        with st.spinner("💬 Generating final answer..."):
            final_response = ask_llm(clarified)
            time.sleep(0.5)

        st.session_state.chat_history.append({
            "original": user_input,
            "clarified": clarified,
            "response": final_response
        })

        st.success("Done!")

# Show conversation
if st.session_state.chat_history:
    st.markdown("### 🗣️ Conversation")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**🔹 You:** {chat['original']}")
        st.markdown(f"**🔍 Clarified:** {chat['clarified']}")
        st.markdown(f"**🤖 Bot:** {chat['response']}")
        st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")
