import streamlit as st
import pandas as pd
import altair as alt

st.title("Traders' Psychology Pyramid")
st.markdown("""
Traders' psychology can be understood as a pyramid, with many beginners at the base and few experts at the peak. Each ascending stage represents a higher level of emotional control, discipline, and trading skill. This model helps visualize the typical progression through the psychological states experienced during the development of a trader.""")

# Define pyramid stages and brief descriptions
stages = [
    {"Stage": "Novice", "Description": "Optimism and excitement about trading, often lacking deep knowledge."},
    {"Stage": "Learner", "Description": "Thrill and confidence as early wins occur; traders may become overconfident."},
    {"Stage": "Struggler", "Description": "Facing anxiety, denial, and fear as losses emerge. Many quit at this stage."},
    {"Stage": "Survivor", "Description": "Gaining risk-awareness and emotional control, learning from mistakes."},
    {"Stage": "Expert", "Description": "Consistent profitability, discipline, and sustainable growth; the few who reach the top."}
]

df = pd.DataFrame(stages)

# Display pyramid data in table
st.subheader("Trading Psychology Pyramid Stages")
st.dataframe(df)

# Prepare data for pyramid visualization (inverted bar chart)
df_viz = pd.DataFrame({
    "Stage": [s["Stage"] for s in stages],
    "Level": [5, 4, 3, 2, 1]  # For pyramid visualization: base = 5, top = 1
})

chart = alt.Chart(df_viz).mark_bar().encode(
    x=alt.X('Stage', sort=None),
    y=alt.Y('Level'),
    color=alt.Color('Stage', legend=None)
).properties(
    title="Traders' Psychology Pyramid (Levels)"
)

st.altair_chart(chart, use_container_width=True)

st.markdown("""
The pyramid starts with optimism and excitement and ascends through stages of challenge, learning, and mastery. This structure illustrates how emotional control and expertise are refined as traders move up.
""")
