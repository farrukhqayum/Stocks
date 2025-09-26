import streamlit as st
from openai import OpenAI
st.write("Secrets:", st.secrets)
# Configure your API key as environment variable or directly here
openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=openai_api_key)

st.title("ML Results Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for ML stock results."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2,
    )
    return response.choices[0].message.content

# Input box
user_input = st.text_input("Ask a question about ML results:")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Call GPT with user input
    answer = chat_with_gpt(user_input)
    
    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
