import streamlit as st
import pandas as pd
import altair as alt

st.title("Traders' Psychology Pyramid")
st.markdown("""
Traders' psychology can be understood as a pyramid, with many beginners at the base and few experts at the peak. Each ascending stage represents a higher level of emotional control, discipline, and trading skill. This model helps visualize the typical progression through the psychological states experienced during the development of a trader.""")

# Define pyramid stages and brief descriptions
stages = [
    {"Stage": "Novice", "Description": "Optimism and excitement about trading, often lacking deep knowledge. Focuses on blaming others for own mistakes."},
    {"Stage": "Learner", "Description": "Thrill and confidence as early wins occur; traders may become overconfident. Consider themselves as experts."},
    {"Stage": "Struggler", "Description": "Facing anxiety, denial, and fear as losses emerge. Totally, confused. Many quit at this stage."},
    {"Stage": "Survivor", "Description": "Gaining risk-awareness and emotional control, learning from mistakes."},
    {"Stage": "Expert", "Description": "Consistent profitability, discipline, and sustainable growth; the few who reach the top."}
]

df = pd.DataFrame({
    "Stage": [s["Stage"] for s in stages],
    "Width": [100, 80, 60, 40, 20],  # widest at bottom, narrowest at top
    "Level": [1, 2, 3, 4, 5]         # Use for stack order
})

pyramid = alt.Chart(df).mark_bar(size=40).encode(
    y=alt.Y('Stage', sort=df["Stage"].tolist()[::-1]),  # reverse for top = Expert
    x=alt.X('Width', scale=alt.Scale(domain=[0, 120]), title=''),
    color=alt.Color('Stage', legend=None)
).properties(
    height=300,
    width=350,
    title="Trading Psychology Pyramid"
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_view(
    stroke=None  # removes chart border
)

st.altair_chart(pyramid, use_container_width=True)

st.markdown("""
The pyramid starts with optimism and excitement and ascends through stages of challenge, learning, and mastery. This structure illustrates how emotional control and expertise are refined as traders move up.
""")
