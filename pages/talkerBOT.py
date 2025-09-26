import streamlit as st
from openai import OpenAI

deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"]
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

st.title("ML Results Chatbot with DeepSeek")

if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_with_deepseek(prompt, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant for ML stock results."},
    ]
    if context:
        messages.append({"role": "system", "content": f"Context: {context}"})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"DeepSeek API error: {e}")
        return ""

# Retrieve ML results and interact similarly
ml_results = st.session_state.get('ml_results', None)
context_str = ml_results.to_string(index=False) if ml_results is not None else ""

user_input = st.text_input("Ask a question about ML results:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    answer = chat_with_deepseek(user_input, context=context_str)
    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages:
    role_label = "You" if msg["role"] == "user" else "Bot"
    st.markdown(f"**{role_label}:** {msg['content']}")
