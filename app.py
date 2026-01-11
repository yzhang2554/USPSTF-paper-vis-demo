import streamlit as st
import pandas as pd
import plotly.express as px
import re
import io


# Load data directly from string to ensure it works immediately
@st.cache_data
def load_data():
    df = pd.read_csv('screening_data.csv')

    def get_years(strategy):
        years = [int(y) for y in re.findall(r'\d+', strategy)]
        return min(years), max(years)

    def get_intervals(strategy):
        intervals = []
        if 'A' in strategy: intervals.append('Annual')
        if 'B' in strategy: intervals.append('Biannual')
        return intervals

    df['Start Year'] = df['Strategy'].apply(lambda x: get_years(x)[0])
    df['End Year'] = df['Strategy'].apply(lambda x: get_years(x)[1])
    df['Intervals'] = df['Strategy'].apply(get_intervals)

    return df


df = load_data()

st.title("USPSTF Screening Paper Breast Cancer Mortality Reduction"
         " Data Explorer")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Options")

model_options = {
    "Median": "Median",
    "Model D": "D",
    "Model E": "E",
    "Model GE": "GE",
    "Model M": "M",
    "Model S": "S",
    "Model W": "W"
}
selected_model_label = st.sidebar.selectbox("Select Output Model", list(model_options.keys()))
selected_col = model_options[selected_model_label]

interval_choice = st.sidebar.selectbox("Screening Interval", ["All", "Annual", "Biannual"])

start_years = ["All"] + sorted([str(y) for y in df['Start Year'].unique()])
start_choice = st.sidebar.selectbox("Start Age", start_years)

end_years = ["All"] + sorted([str(y) for y in df['End Year'].unique()])
end_choice = st.sidebar.selectbox("End Age", end_years)

# --- FILTERING LOGIC ---
filtered_df = df.copy()

# Apply Interval Filter (Check if choice is in the list of intervals for that row)
if interval_choice != "All":
    filtered_df = filtered_df[filtered_df['Intervals'].apply(lambda x: interval_choice in x)]

# Apply Start Year Filter
if start_choice != "All":
    filtered_df = filtered_df[filtered_df['Start Year'] == int(start_choice)]

# Apply End Year Filter
if end_choice != "All":
    filtered_df = filtered_df[filtered_df['End Year'] == int(end_choice)]

# --- VISUALIZATION ---
if not filtered_df.empty:
    # Key Stats

    c1, c2 = st.columns(2)
    c1.metric("Selected Strategies", len(filtered_df))
    c2.metric("Median Screens", f"{filtered_df['Screens'].median():,.0f}")

    # Chart: Screens vs Median Reduction
    st.subheader("Comparison: # of Screens vs Mortality Reduction")
    chart_title = (f"Mortality Reduction (%): {selected_model_label} Results "
                   f"by {interval_choice if interval_choice != "All" else "Annual,Biannual"} interval"
                   )  # f"Start age includes {start_choice} End age includes {end_choice}"
    st.info(
        f"**Note on Color:** The bars are colored by the **Total Number of Screens**. "
        f"Darker/Redder bars indicate a higher total number of screens.")
    fig = px.bar(
        filtered_df,
        x="Strategy",
        y=selected_col,
        color="Screens",
        hover_data=["Screens", selected_col],
        title=chart_title,
        color_continuous_scale="Reds"
    )
    fig.update_layout(xaxis={'categoryorder': 'total ascending', 'tickangle':45},
        yaxis_title="Mortality Reduction (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.write("---")
    st.subheader(f"Alternatives")

    fig_scatter = px.scatter(
        filtered_df,
        x="Screens",
        y=selected_col,
        text="Strategy",  # Shows name near the dot
        hover_name="Strategy",
        title=f"Trade-off: Number of Screens vs. {selected_model_label} Reduction",
        labels={selected_col: "Mortality Reduction (%)", "Screens": "Number of Screens"},
        color_continuous_scale="Viridis"
    )

    # Adjust text position so it doesn't overlap the dots
    fig_scatter.update_traces(textposition='top center')

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.write("---")

    st.info("This chart shows the spread of results across all 6 models (D, E, GE, M, S, W) for each strategy."
            "Hover on each point for tooltips like the source of model. For example, model W and GE predict the most "
            "conservative mortality reduction results.")

    # We "melt" the data to put all model results in one column
    # This allows Plotly to see the distribution for each strategy
    model_cols = ['D', 'E', 'GE', 'M', 'S', 'W']
    df_melted = filtered_df.melt(
        id_vars=['Strategy'],
        value_vars=model_cols,
        var_name='Model',
        value_name='Reduction'
    )

    fig_box = px.box(
        df_melted,
        x="Strategy",
        y="Reduction",
        # color="Strategy",
        title="Distribution of Mortality Reduction Across All Models",
        points="all",  # Shows the individual model dots next to the box
        hover_data=["Model"],
        labels={"Reduction": "Mortality Reduction (%)"}
    )

    fig_box.update_layout(xaxis={'categoryorder': 'total descending', 'tickangle': 45})
    st.plotly_chart(fig_box, use_container_width=True)

    st.write("---")  # Visual divider before the table
    # Table
    st.subheader("Raw Data Output")
    st.dataframe(filtered_df[['Strategy', 'Screens', 'Median', 'D', 'E', 'GE', 'M', 'S', 'W']])
else:
    st.error("No data matches the selected filters. Please try a different combination.")