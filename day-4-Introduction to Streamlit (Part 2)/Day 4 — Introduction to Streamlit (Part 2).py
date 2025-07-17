import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# LLM function
def ask_llm(prompt, model="gpt-4o"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# App title
st.title("🧠 LLM Chatbot")

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Sidebar to reset history
with st.sidebar:
    st.markdown("## 🔁 Options")
    if st.button("Clear Chat History"):
        st.session_state["history"] = []
        st.success("Chat cleared.")

# User input
prompt = st.text_input("Enter your message:")

# When Ask button is clicked
if st.button("Ask"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Waiting for the LLM response..."):
            response = ask_llm(prompt)
        if response.startswith("⚠️"):
            st.error(response)
        else:
            # Save interaction
            st.session_state["history"].append({
                "user": prompt,
                "bot": response
            })
            st.success("Response received!")

# Show conversation history
if st.session_state["history"]:
    st.markdown("### 🗣️ Conversation")
    for chat in st.session_state["history"]:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")  # or use st.divider() if you prefer

# Optional: Debug session state
with st.expander("🧠 Show session state (debug mode)"):
    st.json(st.session_state["history"])

