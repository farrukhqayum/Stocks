import streamlit as st

st.sidebar.title("Navigation Menu")

# Define the pages and their order
pages = ["Home", "About_This_App", "Model Results", "compounding", "MONTHLY_GOAL"]

# User selects page from ordered list
selected_page = st.sidebar.radio("Go to", pages)

# Display page content based on selection
if selected_page == "Home":
    st.title("Home Page")
elif selected_page == "Data Analysis":
    st.title("Data Analysis Page")
elif selected_page == "Model Results":
    st.title("Model Results Page")
elif selected_page == "Settings":
    st.title("Settings Page")
elif selected_page == "About":
    st.title("About Page")

