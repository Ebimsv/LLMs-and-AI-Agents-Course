"""
Complete Streamlit LLM Chat Application
======================================

This file contains a complete, production-ready Streamlit application for building
LLM chat interfaces. Each section is thoroughly explained with comments.

To run this app:
1. Save this file as app.py
2. Install dependencies: pip install streamlit openai anthropic requests python-dotenv
3. Run: streamlit run app.py
4. Open browser to http://localhost:8501

Author: LLMs and Agents in Production Tutorial Series
Date: 2025
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import streamlit as st
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This allows you to store API keys securely
load_dotenv()

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

# st.set_page_config() MUST be the first Streamlit command
# This configures the page appearance and behavior
st.set_page_config(
    page_title="LLM Chat Assistant",  # Browser tab title
    page_icon="🧠",                   # Browser tab icon (emoji or file path)
    layout="wide",                     # "wide" or "centered"
    initial_sidebar_state="expanded"   # "expanded" or "collapsed"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

# Custom CSS for better visual appearance
# This is injected into the HTML head
st.markdown("""
<style>
/* Custom styling for chat messages */
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}

.chat-message.user {
    background-color: #2b313e;
}

.chat-message.assistant {
    background-color: #475063;
}

.chat-message .avatar {
    width: 20%;
}

.chat-message .message {
    width: 80%;
}

/* Custom styling for metrics */
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

/* Custom styling for buttons */
.stButton > button {
    width: 100%;
    border-radius: 0.5rem;
}

/* Custom styling for text inputs */
.stTextInput > div > div > input {
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Session state persists data across app reruns
# This is crucial for maintaining chat history and app state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# =============================================================================
# SIDEBAR - SETTINGS AND CONTROLS
# =============================================================================

# st.sidebar creates a sidebar on the left side of the app
# All widgets inside st.sidebar appear in the sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Provider selection using st.selectbox()
    # st.selectbox() creates a dropdown menu
    # Parameters:
    # - label: Text displayed above the dropdown
    # - options: List of options to choose from
    # - index: Default selected option (0-based)
    # - help: Tooltip text shown on hover
    provider = st.selectbox(
        "Choose Provider",
        ["OpenAI", "Ollama", "Anthropic"],
        help="Select your LLM provider"
    )
    
    # Model selection based on provider
    # This demonstrates conditional logic in Streamlit
    if provider == "OpenAI":
        model = st.selectbox(
            "Choose Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            help="Select OpenAI model"
        )
    elif provider == "Ollama":
        model = st.selectbox(
            "Choose Model",
            ["llama2", "codellama", "mistral", "neural-chat"],
            help="Select Ollama model"
        )
    else:  # Anthropic
        model = st.selectbox(
            "Choose Model",
            ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"],
            help="Select Anthropic model"
        )
    
    # Model parameters section
    st.subheader("Model Parameters")
    
    # st.slider() creates a slider input
    # Parameters:
    # - label: Text displayed above the slider
    # - min_value: Minimum value
    # - max_value: Maximum value
    # - value: Default value
    # - step: Step size
    # - help: Tooltip text
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses (0 = deterministic, 2 = very random)"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum length of response"
    )
    
    # API Configuration section
    st.subheader("API Configuration")
    
    # Conditional API key input based on provider
    if provider == "OpenAI":
        # st.text_input() with type="password" creates a password field
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",  # Hides the input
            value=os.getenv("OPENAI_API_KEY", ""),  # Default from environment
            help="Enter your OpenAI API key"
        )
    elif provider == "Anthropic":
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            help="Enter your Anthropic API key"
        )
    else:  # Ollama
        api_key = ""  # No API key needed for local Ollama
        # st.info() displays an informational message
        st.info("No API key needed for local Ollama")
    
    # Chat controls section
    st.subheader("Chat Controls")
    
    # st.button() creates a clickable button
    # Parameters:
    # - label: Text displayed on the button
    # - type: "primary" or "secondary" (affects styling)
    # - help: Tooltip text
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        # st.rerun() reruns the entire app
        st.rerun()
    
    if st.button("📥 Export Chat", type="secondary"):
        # Export chat history as JSON
        chat_data = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "messages": st.session_state.messages
        }
        
        # st.download_button() creates a download button
        # Parameters:
        # - label: Text displayed on the button
        # - data: Data to download
        # - file_name: Name of downloaded file
        # - mime: MIME type of the file
        st.download_button(
            label="Download Chat History",
            data=json.dumps(chat_data, indent=2),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

# st.title() creates a large, prominent heading
st.title("🧠 LLM Chat Assistant")

# st.markdown() renders markdown text
# "---" creates a horizontal rule
st.markdown("---")

# =============================================================================
# DISPLAY CHAT MESSAGES
# =============================================================================

# Display existing chat messages
for message in st.session_state.messages:
    # st.chat_message() creates a chat message container
    # Parameters:
    # - name: Role name (user, assistant, etc.)
    # - avatar: Custom avatar (emoji, image, etc.)
    with st.chat_message(message["role"]):
        # st.markdown() renders the message content
        st.markdown(message["content"])
        
        # Display timestamp if available
        if "timestamp" in message:
            # st.caption() displays small text
            st.caption(f"{message['timestamp']}")

# =============================================================================
# CHAT INPUT AND RESPONSE GENERATION
# =============================================================================

# st.chat_input() creates a chat input field
# This is specifically designed for chat interfaces
# Parameters:
# - placeholder: Text shown when input is empty
# - key: Unique identifier for this widget
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        # st.empty() creates a placeholder for dynamic content
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # st.spinner() shows a loading spinner
            # This provides visual feedback during long operations
            with st.spinner("Thinking..."):
                if provider == "OpenAI":
                    # OpenAI API integration
                    import openai
                    client = openai.OpenAI(api_key=api_key)
                    
                    # Create chat completion with streaming
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True  # Enable streaming for real-time display
                    )
                    
                    # Process streaming response
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            # Update placeholder with current response + cursor
                            message_placeholder.markdown(full_response + "▌")
                    
                    # Final update without cursor
                    message_placeholder.markdown(full_response)
                    
                elif provider == "Ollama":
                    # Ollama API integration (local LLM)
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": True,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens
                            }
                        }
                    )
                    
                    # Process streaming response from Ollama
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                full_response += data["response"]
                                message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                else:  # Anthropic
                    # Anthropic API integration
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)
                    
                    response = client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    full_response = response.content[0].text
                    message_placeholder.markdown(full_response)
                
        except Exception as e:
            # st.error() displays an error message
            st.error(f"Error: {str(e)}")
            full_response = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

# =============================================================================
# FOOTER AND STATISTICS
# =============================================================================

# Footer with app statistics
st.markdown("---")

# Display statistics using HTML for better formatting
st.markdown(
    f"""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Using {provider} with {model} • 
        Messages: {len(st.session_state.messages)} • 
        Temperature: {temperature} • 
        Max Tokens: {max_tokens}
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# ADDITIONAL FEATURES AND EXAMPLES
# =============================================================================

# Example of using st.columns() for layout
# This creates a multi-column layout
col1, col2, col3 = st.columns(3)

with col1:
    # st.metric() displays key metrics
    st.metric("Total Messages", len(st.session_state.messages))

with col2:
    st.metric("User Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))

with col3:
    st.metric("Assistant Messages", len([m for m in st.session_state.messages if m["role"] == "assistant"]))

# Example of using st.expander() for collapsible sections
with st.expander("📊 Chat Statistics", expanded=False):
    if st.session_state.messages:
        # Calculate average message length
        avg_length = sum(len(m["content"]) for m in st.session_state.messages) / len(st.session_state.messages)
        st.write(f"Average message length: {avg_length:.1f} characters")
        
        # Show most recent messages
        st.write("Recent messages:")
        for msg in st.session_state.messages[-3:]:
            st.write(f"- {msg['role']}: {msg['content'][:50]}...")

# Example of using st.tabs() for organized content
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Analytics", "⚙️ Settings"])

with tab1:
    st.write("This is the main chat interface.")

with tab2:
    st.write("Analytics and insights would go here.")

with tab3:
    st.write("Additional settings and configuration options.")

# =============================================================================
# IMPORTANT STREAMLIT COMMANDS REFERENCE
# =============================================================================

"""
IMPORTANT STREAMLIT COMMANDS REFERENCE
=====================================

📝 TEXT AND DISPLAY:
st.write()           - Display any data type
st.text()            - Display plain text
st.markdown()        - Display markdown text
st.latex()           - Display LaTeX math
st.code()            - Display code with syntax highlighting
st.json()            - Display JSON data

📊 DATA DISPLAY:
st.dataframe()       - Display pandas DataFrame
st.table()           - Display static table
st.metric()          - Display key metrics
st.line_chart()      - Line chart
st.bar_chart()       - Bar chart
st.area_chart()      - Area chart
st.scatter_chart()   - Scatter plot
st.map()             - Display map
st.plotly_chart()    - Display Plotly charts
st.altair_chart()    - Display Altair charts

🎛️ INPUT WIDGETS:
st.text_input()      - Single line text input
st.text_area()       - Multi-line text input
st.number_input()    - Number input
st.selectbox()       - Dropdown selection
st.multiselect()     - Multi-selection dropdown
st.slider()          - Slider input
st.select_slider()   - Select from range
st.checkbox()        - Checkbox
st.radio()           - Radio buttons
st.button()          - Button
st.download_button() - Download button
st.file_uploader()   - File upload
st.camera_input()    - Camera input
st.color_picker()    - Color picker
st.date_input()      - Date input
st.time_input()      - Time input

📱 LAYOUT:
st.sidebar           - Sidebar container
st.columns()         - Multi-column layout
st.container()       - Container for grouping
st.expander()        - Collapsible section
st.tabs()            - Tabbed interface
st.empty()           - Placeholder for dynamic content

💬 CHAT INTERFACE:
st.chat_input()      - Chat input field
st.chat_message()    - Chat message container

📊 CHARTS AND PLOTS:
st.line_chart()      - Line chart
st.bar_chart()       - Bar chart
st.area_chart()      - Area chart
st.scatter_chart()   - Scatter plot
st.map()             - Map display
st.plotly_chart()    - Plotly integration
st.altair_chart()    - Altair integration
st.vega_lite_chart() - Vega-Lite charts

⚙️ CONFIGURATION:
st.set_page_config() - Page configuration
st.experimental_memo - Cache function results
st.cache_data        - Cache data loading
st.cache_resource    - Cache resources

🔄 STATE MANAGEMENT:
st.session_state     - Persistent state across reruns
st.rerun()           - Rerun the app
st.stop()            - Stop execution

📱 MOBILE AND RESPONSIVE:
st.set_page_config(layout="wide")  - Wide layout
st.set_page_config(layout="centered")  - Centered layout

🎨 STYLING:
st.markdown() with HTML - Custom styling
st.components.html() - Custom HTML components
st.components.iframe() - Embed external content

📊 PROGRESS AND STATUS:
st.progress()        - Progress bar
st.spinner()         - Loading spinner
st.balloons()        - Celebration balloons
st.snow()            - Snow effect

💬 MESSAGES:
st.success()         - Success message
st.error()           - Error message
st.warning()         - Warning message
st.info()            - Info message
st.exception()       - Display exception

📁 FILE HANDLING:
st.file_uploader()   - Upload files
st.download_button() - Download files

🔧 FORMS:
st.form()            - Form container
st.form_submit_button() - Form submit button

📊 CACHING:
@st.cache_data       - Cache data loading functions
@st.cache_resource   - Cache resource loading
@st.experimental_memo - Cache function results

🎯 BEST PRACTICES:
- Use st.session_state for persistent data
- Cache expensive operations with @st.cache_data
- Use st.empty() for dynamic content updates
- Group related widgets in containers
- Use st.sidebar for settings and controls
- Implement proper error handling
- Use st.spinner() for long operations
- Optimize for mobile with responsive layouts
"""

# =============================================================================
# RUNNING THE APP
# =============================================================================

"""
TO RUN THIS APP:
================

1. Save this file as app.py
2. Install dependencies:
   pip install streamlit openai anthropic requests python-dotenv

3. Run the app:
   streamlit run app.py

4. Open browser to http://localhost:8501

ADDITIONAL COMMANDS:
===================

- streamlit run app.py --server.port 8502  # Custom port
- streamlit run app.py --server.address 0.0.0.0  # Allow external access
- streamlit run app.py --server.headless true  # Headless mode
- streamlit run app.py --browser.gatherUsageStats false  # Disable usage stats

DEVELOPMENT TIPS:
================

- Use --server.runOnSave true for auto-reload
- Use --server.enableCORS false for local development
- Use --server.enableXsrfProtection false for testing
- Use st.write() for debugging
- Use st.session_state to persist data
- Use st.empty() for dynamic updates
- Use st.spinner() for loading states
- Use st.error() for error handling

DEPLOYMENT OPTIONS:
==================

🌐 Streamlit Cloud (Recommended):
- Free hosting for public repos
- Automatic deployment from GitHub
- Easy sharing and collaboration
- Built-in CI/CD

🐳 Docker:
- Containerized deployment
- Consistent environment
- Easy scaling
- Cloud platform support

☁️ Cloud Platforms:
- Heroku
- Google Cloud Run
- AWS App Runner
- Azure Container Instances

🖥️ Self-hosted:
- VPS deployment
- Custom domain
- Full control
- Cost-effective for high traffic

📱 Mobile:
- Streamlit apps work on mobile
- Responsive design recommended
- Touch-friendly interfaces
""" 