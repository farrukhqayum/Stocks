import streamlit as st
import pandas as pd
import altair as alt

st.title("Traders' Psychology Pyramid")

stages = [
    {"Stage": "Novice", "Description": "Optimism and excitement about trading, often lacking deep knowledge. Focuses on blaming others for own mistakes."},
    {"Stage": "Learner", "Description": "Thrill and confidence as early wins occur; traders may become overconfident. Consider themselves as experts."},
    {"Stage": "Struggler", "Description": "Facing anxiety, denial, and fear as losses emerge. Totally, confused. Many quit at this stage."},
    {"Stage": "Survivor", "Description": "Gaining risk-awareness and emotional control, learning from mistakes."},
    {"Stage": "Expert", "Description": "Consistent profitability, discipline, and sustainable growth; the few who reach the top."}
]

df = pd.DataFrame({
    "Stage": [s["Stage"] for s in reversed(stages)],  # Reverse for pyramid order (base at bottom)
    "Level": [5, 4, 3, 2, 1]  # Pyramid levels, base = 5, apex = 1
})

# Display pyramid data in table
st.subheader("Trading Psychology Pyramid Stages")
st.dataframe(pd.DataFrame(stages))

# Create a pyramid-like horizontal bar chart
bar = alt.Chart(df).mark_bar().encode(
    y=alt.Y('Stage:N', sort=None),
    x=alt.X('Level:Q', scale=alt.Scale(domain=[0, 5]), title='Level'),
    color=alt.Color('Stage:N', legend=None)
).properties(
    height=300,
    width=350,
    title="Traders' Psychology Pyramid"
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
)

st.altair_chart(bar, use_container_width=True)

st.markdown("""
The horizontal bars mirror the shape of a pyramid—wider at the bottom and narrower at the top—helping visualize the journey from novice to expert in trading psychology.
""")
