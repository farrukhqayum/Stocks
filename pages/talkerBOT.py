import streamlit as st
from openai import OpenAI

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

st.title("ML Results Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_with_gpt(prompt, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant for ML stock results."},
    ]
    # Add context if any
    if context:
        messages.append({"role": "system", "content": f"Context: {context}"})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        temperature=0.2,
    )
    return response.choices[0].message.content

# Get stored ML results from session_state
ml_results = st.session_state.get('ml_results', None)
if ml_results is not None:
    st.write("Loaded ML results:")
    st.dataframe(ml_results)
    # Create a simple plain text representation of the results for GPT context
    context_str = ml_results.to_string(index=False)
else:
    st.warning("ML results not found. Run the main page first.")
    context_str = ""

user_input = st.text_input("Ask a question about ML results:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    answer = chat_with_gpt(user_input, context=context_str)
    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
