import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(
    page_title='Workforce Productivity Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv('time_productivity_data.csv')
    # Data type conversions
    df['User ID'] = df['User ID'].astype(str)
    df['Age'] = df['Age'].astype(int)
    df['Daily Work Hours'] = df['Daily Work Hours'].astype(float)
    df['Daily Leisure Hours'] = df['Daily Leisure Hours'].astype(float)
    df['Daily Exercise Minutes'] = df['Daily Exercise Minutes'].astype(int)
    df['Daily Sleep Hours'] = df['Daily Sleep Hours'].astype(float)
    df['Productivity Score'] = df['Productivity Score'].astype(int)
    df['Screen Time (hours)'] = df['Screen Time (hours)'].astype(float)
    df['Commute Time (hours)'] = df['Commute Time (hours)'].astype(float)
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title('Studies Navigation')

# Create a static list of studies
studies = [
    'Dashboard',
    '1. Work Hours vs. Productivity',
    '2. Leisure Hours vs. Productivity',
    '3. Exercise Minutes vs. Productivity',
    '4. Sleep Hours vs. Productivity',
    '5. Screen Time vs. Productivity',
    '6. Commute Time vs. Productivity',
    '7. Age vs. Productivity',
    '8. Work-Life Balance vs. Productivity',
    '9. Predicting Productivity',
    '10. Clustering Employees'
]

# Display studies as a static list using radio buttons
selection = st.sidebar.radio('Select a Study', studies)

# Default Dashboard
if selection == 'Dashboard':
    st.header('Overall Productivity Analysis Dashboard')

    # Summary Statistics
    st.subheader('Summary Statistics')
    summary_cols = ['Productivity Score', 'Daily Work Hours', 'Daily Leisure Hours', 'Daily Sleep Hours', 'Daily Exercise Minutes', 'Screen Time (hours)', 'Commute Time (hours)']
    summary_df = df[summary_cols].describe().transpose()
    st.dataframe(summary_df.style.highlight_max(axis=0, props="background-color: yellow; color: black"))

    # Correlation Matrix
    st.subheader('Correlation Matrix')
    corr = df[summary_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect='auto',
        color_continuous_scale='RdBu_r',
        origin='lower',
        title='Correlation Matrix'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Key Insights
    st.subheader('Key Insights')
    avg_productivity = df['Productivity Score'].mean()
    avg_work_hours = df['Daily Work Hours'].mean()
    avg_leisure_hours = df['Daily Leisure Hours'].mean()
    avg_sleep_hours = df['Daily Sleep Hours'].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Average Productivity Score", value=f"{avg_productivity:.2f}")

    with col2:
        st.metric(label="Average Daily Work Hours", value=f"{avg_work_hours:.2f} hrs")

    with col3:
        st.metric(label="Average Daily Leisure Hours", value=f"{avg_leisure_hours:.2f} hrs")

    with col4:
        st.metric(label="Average Daily Sleep Hours", value=f"{avg_sleep_hours:.2f} hrs")

    st.markdown("""
    **Correlation Highlights:**
    - **Productivity Score** has a strong positive correlation with **Daily Work Hours** and **Daily Sleep Hours**.
    - **Productivity Score** has a negative correlation with **Screen Time (hours)**.
    - These insights suggest that optimizing work hours and sleep can enhance productivity, while excessive screen time may hinder it.
    
    Use the sidebar to navigate through detailed studies for more in-depth analysis.
    """)

# Define studies
elif selection == '1. Work Hours vs. Productivity':
    st.header('Study 1: Daily Work Hours vs. Productivity Score')

    # Scatter plot with Productivity Score on Y-axis
    fig = px.scatter(
        df, x='Daily Work Hours', y='Productivity Score', trendline='ols',
        labels={
            'Daily Work Hours': 'Daily Work Hours (hours)',
            'Productivity Score': 'Productivity Score'
        },
        title='Daily Work Hours vs. Productivity Score',
        template='plotly_white'
    )
    fig.update_layout(width=700, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.markdown("""
    **Insights:**
    - Examines the relationship between daily work hours and productivity.
    - Helps identify if longer work hours correlate with higher productivity.
    """)

    # Statistical analysis
    results = px.get_trendline_results(fig)
    model = results.iloc[0]["px_fit_results"]
    st.subheader('Statistical Analysis:')
    st.write(f'- **Slope (Coefficient):** {model.params[1]:.4f}')
    st.write(f'- **Intercept:** {model.params[0]:.4f}')
    st.write(f'- **R-squared:** {model.rsquared:.4f}')
    st.write('- **Interpretation:** A positive slope indicates that increased work hours are associated with higher productivity scores.')

elif selection == '2. Leisure Hours vs. Productivity':
    st.header('Study 2: Daily Leisure Hours vs. Productivity Score')

    fig = px.scatter(
        df, x='Daily Leisure Hours', y='Productivity Score', trendline='ols',
        labels={
            'Daily Leisure Hours': 'Daily Leisure Hours (hours)',
            'Productivity Score': 'Productivity Score'
        },
        title='Daily Leisure Hours vs. Productivity Score',
        template='plotly_white'
    )
    fig.update_layout(width=700, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insights:**
    - Analyzes how leisure time affects productivity.
    - Determines if more leisure time contributes to higher productivity.
    """)

    results = px.get_trendline_results(fig)
    model = results.iloc[0]["px_fit_results"]
    st.subheader('Statistical Analysis:')
    st.write(f'- **Slope (Coefficient):** {model.params[1]:.4f}')
    st.write(f'- **Intercept:** {model.params[0]:.4f}')
    st.write(f'- **R-squared:** {model.rsquared:.4f}')
    st.write('- **Interpretation:** A positive slope suggests that increased leisure time is associated with higher productivity scores.')

elif selection == '3. Exercise Minutes vs. Productivity':
    st.header('Study 3: Daily Exercise Minutes vs. Productivity Score')

    fig = px.scatter(
        df, x='Daily Exercise Minutes', y='Productivity Score', trendline='ols',
        labels={
            'Daily Exercise Minutes': 'Daily Exercise Minutes (minutes)',
            'Productivity Score': 'Productivity Score'
        },
        title='Daily Exercise Minutes vs. Productivity Score',
        template='plotly_white'
    )
    fig.update_layout(width=700, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insights:**
    - Assesses whether exercise duration influences productivity.
    - Identifies optimal exercise time for productivity.
    """)

    results = px.get_trendline_results(fig)
    model = results.iloc[0]["px_fit_results"]
    st.subheader('Statistical Analysis:')
    st.write(f'- **Slope (Coefficient):** {model.params[1]:.4f}')
    st.write(f'- **Intercept:** {model.params[0]:.4f}')
    st.write(f'- **R-squared:** {model.rsquared:.4f}')
    st.write('- **Interpretation:** A positive slope indicates that more exercise minutes are associated with higher productivity scores.')

elif selection == '4. Sleep Hours vs. Productivity':
    st.header('Study 4: Daily Sleep Hours vs. Productivity Score')

    fig = px.scatter(
        df, x='Daily Sleep Hours', y='Productivity Score', trendline='ols',
        labels={
            'Daily Sleep Hours': 'Daily Sleep Hours (hours)',
            'Productivity Score': 'Productivity Score'
        },
        title='Daily Sleep Hours vs. Productivity Score',
        template='plotly_white'
    )
    fig.update_layout(width=700, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insights:**
    - Investigates how sleep duration affects productivity.
    - Helps identify optimal sleep hours for maximum productivity.
    """)

    results = px.get_trendline_results(fig)
    model = results.iloc[0]["px_fit_results"]
    st.subheader('Statistical Analysis:')
    st.write(f'- **Slope (Coefficient):** {model.params[1]:.4f}')
    st.write(f'- **Intercept:** {model.params[0]:.4f}')
    st.write(f'- **R-squared:** {model.rsquared:.4f}')
    st.write('- **Interpretation:** A positive slope suggests that more sleep hours are associated with higher productivity scores.')

elif selection == '5. Screen Time vs. Productivity':
    st.header('Study 5: Screen Time vs. Productivity Score')

    fig = px.scatter(
        df, x='Screen Time (hours)', y='Productivity Score', trendline='ols',
        labels={
            'Screen Time (hours)': 'Screen Time (hours)',
            'Productivity Score': 'Productivity Score'
        },
        title='Screen Time vs. Productivity Score',
        template='plotly_white'
    )
    fig.update_layout(width=700, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insights:**
    - Evaluates how screen time correlates with productivity.
    - Determines if excessive screen time affects productivity negatively.
    """)

    results = px.get_trendline_results(fig)
    model = results.iloc[0]["px_fit_results"]
    st.subheader('Statistical Analysis:')
    st.write(f'- **Slope (Coefficient):** {model.params[1]:.4f}')
    st.write(f'- **Intercept:** {model.params[0]:.4f}')
    st.write(f'- **R-squared:** {model.rsquared:.4f}')
    st.write('- **Interpretation:** A negative slope would indicate that increased screen time is associated with lower productivity scores.')

elif selection == '6. Commute Time vs. Productivity':
    st.header('Study 6: Commute Time vs. Productivity Score')

    fig = px.scatter(
        df, x='Commute Time (hours)', y='Productivity Score', trendline='ols',
        labels={
            'Commute Time (hours)': 'Commute Time (hours)',
            'Productivity Score': 'Productivity Score'
        },
        title='Commute Time vs. Productivity Score',
        template='plotly_white'
    )
    fig.update_layout(width=700, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insights:**
    - Analyzes the impact of commute duration on productivity.
    - Considers if remote work policies might improve productivity.
    """)

    results = px.get_trendline_results(fig)
    model = results.iloc[0]["px_fit_results"]
    st.subheader('Statistical Analysis:')
    st.write(f'- **Slope (Coefficient):** {model.params[1]:.4f}')
    st.write(f'- **Intercept:** {model.params[0]:.4f}')
    st.write(f'- **R-squared:** {model.rsquared:.4f}')
    st.write('- **Interpretation:** A negative slope suggests that longer commute times are associated with lower productivity scores.')

elif selection == '7. Age vs. Productivity':
    st.header('Study 7: Age vs. Productivity Score')

    # Create age groups
    df['Age Group'] = pd.cut(
        df['Age'],
        bins=[20, 30, 40, 50, 60, 70],
        labels=['21-30', '31-40', '41-50', '51-60', '61-70']
    )

    fig = px.box(
        df, x='Age Group', y='Productivity Score',
        labels={
            'Age Group': 'Age Group',
            'Productivity Score': 'Productivity Score'
        },
        title='Productivity Score by Age Group',
        color='Age Group',
        template='plotly_white'
    )
    fig.update_layout(width=700, height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insights:**
    - Compares productivity scores across different age groups.
    - Identifies age demographics with higher productivity.
    """)

    group_means = df.groupby('Age Group')['Productivity Score'].mean().reset_index()
    st.subheader('Average Productivity Score by Age Group:')
    st.table(
        group_means.style
        .highlight_max(axis=0, props="background-color: yellow; color: black")
    )

elif selection == '8. Work-Life Balance vs. Productivity':
    st.header('Study 8: Work-Life Balance vs. Productivity Score')

    # Calculate work-life balance ratio
    df['Work-Life Ratio'] = df['Daily Work Hours'] / (df['Daily Leisure Hours'] + 1e-5)

    fig = px.scatter(
        df, x='Work-Life Ratio', y='Productivity Score', trendline='ols',
        labels={
            'Work-Life Ratio': 'Work-Life Ratio',
            'Productivity Score': 'Productivity Score'
        },
        title='Work-Life Balance vs. Productivity Score',
        template='plotly_white'
    )
    fig.update_layout(width=700, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insights:**
    - Examines the effect of work-life balance on productivity.
    - A lower ratio indicates a better balance.
    """)

    results = px.get_trendline_results(fig)
    model = results.iloc[0]['px_fit_results']
    st.subheader('Statistical Analysis:')
    st.write(f'- **Slope (Coefficient):** {model.params[1]:.4f}')
    st.write(f'- **Intercept:** {model.params[0]:.4f}')
    st.write(f'- **R-squared:** {model.rsquared:.4f}')
    st.write('- **Interpretation:** A positive slope suggests that a higher work-life ratio (more work compared to leisure) is associated with higher productivity, which may warrant further investigation.')

elif selection == '9. Predicting Productivity':
    st.header('Study 9: Multivariate Analysis to Predict Productivity Score')

    # Select features
    features = [
        'Daily Work Hours',
        'Daily Leisure Hours',
        'Daily Exercise Minutes',
        'Daily Sleep Hours',
        'Screen Time (hours)',
        'Commute Time (hours)'
    ]
    X = df[features]
    y = df['Productivity Score']

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Display coefficients
    coeff_df = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
    st.subheader('Regression Coefficients:')
    st.table(
        coeff_df.style
        .highlight_max(axis=0, props="background-color: yellow; color: black")
    )

    # Display R-squared
    r_squared = model.score(X, y)
    st.write(f'**R-squared:** {r_squared:.4f}')

    # Insights
    st.markdown("""
    **Insights:**
    - Determines which factors most influence productivity.
    - Higher absolute coefficient values indicate stronger influence.
    - R-squared indicates how well the model explains the variability in productivity.
    """)

elif selection == '10. Clustering Employees':
    st.header('Study 10: Clustering Employees Based on Lifestyle Patterns')

    # Select features for clustering
    clustering_features = [
        'Daily Work Hours',
        'Daily Leisure Hours',
        'Daily Exercise Minutes',
        'Daily Sleep Hours'
    ]
    X = df[clustering_features]

    # Perform clustering with increased number of clusters (e.g., 5)
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Visualize clusters in 3D
    fig = px.scatter_3d(
        df,
        x='Daily Work Hours',
        y='Daily Leisure Hours',
        z='Daily Sleep Hours',
        color='Cluster',
        labels={
            'Daily Work Hours': 'Daily Work Hours (hours)',
            'Daily Leisure Hours': 'Daily Leisure Hours (hours)',
            'Daily Sleep Hours': 'Daily Sleep Hours (hours)'
        },
        title=f'Employee Clusters Based on Lifestyle Patterns (k={num_clusters})',
        template='plotly_white'
    )
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Display average productivity per cluster
    cluster_productivity = df.groupby('Cluster')['Productivity Score'].mean().reset_index()
    st.subheader('Average Productivity Score by Cluster:')
    st.table(
        cluster_productivity.style
        .format({'Productivity Score': '{:.2f}'})
        .highlight_max(axis=0, props="background-color: yellow; color: black")
    )

    st.markdown("""
    **Insights:**
    - Segments employees into distinct clusters based on their lifestyle habits.
    - Identifies clusters with varying productivity scores.
    - Helps in tailoring strategies for different employee segments to enhance overall productivity.
    """)

else:
    st.write("Select a study from the sidebar to view the analysis.")

# --- Persistent Footer ---
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: black;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
.footer a {
    color: #007bff;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
<div class="footer">
    Data Source: <a href="https://www.kaggle.com/datasets/hanaksoy/time-management-and-productivity-insights" target="_blank">Kaggle: Time Management and Productivity Insights</a>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
