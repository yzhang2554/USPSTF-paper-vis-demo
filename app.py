import streamlit as st
import pandas as pd
import plotly.express as px
import re


# Load and clean data
@st.cache_data
def load_data():
    # Replace with your actual file path
    df = pd.read_csv('complete_screening_data.csv')

    # 1. Rename Mammography types to shortcuts
    name_map = {
        "Digital mammography": "DM",
        "Digital breast tomosynthesis": "DBT"
    }
    df['Mammography type'] = df['Mammography type'].map(lambda x: name_map.get(x.strip(), x))

    # 2. Data Cleaning: Convert 'n/a' to actual NaN (numeric)
    model_cols = ['D', 'E', 'GE', 'M', 'S', 'W', 'Median']
    for col in model_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Handle Model S requirement:
    # If Type is DM, we force Model S to NaN so it's excluded from calculations/plots
    df.loc[df['Mammography type'] == 'DM', 'S'] = None

    def get_years(strategy):
        years = [int(y) for y in re.findall(r'\d+', str(strategy))]
        if not years: return 0, 0
        return min(years), max(years)

    def get_intervals(strategy):
        intervals = []
        strategy_str = str(strategy)
        if 'A' in strategy_str: intervals.append('Annual')
        if 'B' in strategy_str: intervals.append('Biennial')
        return intervals

    df['Start Year'] = df['Strategy'].apply(lambda x: get_years(x)[0])
    df['End Year'] = df['Strategy'].apply(lambda x: get_years(x)[1])
    df['Intervals'] = df['Strategy'].apply(get_intervals)

    return df


df = load_data()

st.title("USPSTF Screening Data Explorer")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Options")

# Mammography Type Filter (Now using DM/DBT)
mammography_options = sorted(df['Mammography type'].unique().tolist())
selected_mammography = st.sidebar.selectbox("Mammography Type", mammography_options)

# Outcome Filter
outcome_options = sorted(df['Outcome'].unique().tolist())
selected_outcome = st.sidebar.selectbox("Projected Outcome", outcome_options)

st.sidebar.markdown("---")

# Model Selection - Filter out Model S if DM is selected
model_options = {
    "Median": "Median",
    "Model D": "D",
    "Model E": "E",
    "Model GE": "GE",
    "Model M": "M",
    "Model W": "W"
}
# Only add Model S to options if the selected type is NOT DM
if selected_mammography != "DM":
    model_options["Model S"] = "S"

selected_model_label = st.sidebar.selectbox("Select Model for Display", list(model_options.keys()))
selected_col = model_options[selected_model_label]

# Screening Parameters
interval_choice = st.sidebar.selectbox("Screening Interval", ["All", "Annual", "Biennial"])
start_years = ["All"] + sorted([str(y) for y in df['Start Year'].unique()])
start_choice = st.sidebar.selectbox("Start Age", start_years)
end_years = ["All"] + sorted([str(y) for y in df['End Year'].unique()])
end_choice = st.sidebar.selectbox("End Age", end_years)

# --- FILTERING LOGIC ---
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df['Mammography type'] == selected_mammography]
filtered_df = filtered_df[filtered_df['Outcome'] == selected_outcome]

if interval_choice != "All":
    filtered_df = filtered_df[filtered_df['Intervals'].apply(lambda x: interval_choice in x)]
if start_choice != "All":
    filtered_df = filtered_df[filtered_df['Start Year'] == int(start_choice)]
if end_choice != "All":
    filtered_df = filtered_df[filtered_df['End Year'] == int(end_choice)]

# --- VISUALIZATION ---
if not filtered_df.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Strategies", len(filtered_df))
    c2.metric("Median Screens", f"{filtered_df['Screens'].median():,.0f}")
    c3.metric("Type", selected_mammography)

    # 1. Bar Chart
    st.subheader(f"{selected_outcome} Analysis")
    fig = px.bar(
        filtered_df,
        x="Strategy",
        y=selected_col,
        color="Screens",
        title=f"{selected_outcome}: {selected_model_label} ({selected_mammography})",
        color_continuous_scale="Reds"
    )
    fig.update_layout(xaxis={'categoryorder': 'total ascending', 'tickangle': 45})
    st.plotly_chart(fig, use_container_width=True)

    # 2. Scatter Plot
    st.write("---")
    fig_scatter = px.scatter(
        filtered_df,
        x="Screens",
        y=selected_col,
        text="Strategy",
        hover_name="Strategy",
        title=f"Trade-off: Screens vs. {selected_outcome}",
        labels={selected_col: selected_outcome, "Screens": "Number of Screens"},
    )
    fig_scatter.update_traces(textposition='top center')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Box Plot (Distribution of all available models)
    st.write("---")
    st.subheader("Model Distribution")

    # Dynamically determine which model columns to include in the box plot
    current_model_cols = ['D', 'E', 'GE', 'M', 'W']
    if selected_mammography != "DM":
        current_model_cols.append('S')

    df_melted = filtered_df.melt(
        id_vars=['Strategy'],
        value_vars=current_model_cols,
        var_name='Model',
        value_name='Val'
    )

    fig_box = px.box(
        df_melted,
        x="Strategy",
        y="Val",
        title=f"Spread of Models for {selected_mammography}",
        points="all",
        labels={"Val": selected_outcome}
    )
    fig_box.update_layout(xaxis={'categoryorder': 'total descending', 'tickangle': 45})
    st.plotly_chart(fig_box, use_container_width=True)

    # Table
    st.write("---")
    cols_to_show = ['Mammography type', 'Outcome', 'Strategy', 'Screens', 'Median', 'D', 'E', 'GE', 'M', 'W']
    if selected_mammography != "DM":
        cols_to_show.append('S')
    st.dataframe(filtered_df[cols_to_show])

else:
    st.error("No data matches the filters.")