Workforce Productivity Dashboard
This project is a Streamlit application that analyzes workforce productivity based on various factors such as work hours, leisure time, exercise, sleep, screen time, commute time, and age. The app provides interactive visualizations and statistical analyses to help understand how these factors influence productivity.

Table of Contents
Introduction
Features
Data Source
Installation
Usage
Application Structure
Dependencies
Contributing
License
Introduction
Understanding the factors that influence employee productivity is crucial for organizations aiming to improve performance and employee well-being. This application provides insights into how various lifestyle and work-related factors affect productivity, enabling data-driven decisions.

Features
Interactive Visualizations: Scatter plots, box plots, and 3D scatter plots to explore relationships between variables.
Statistical Analysis: Linear regression analyses with coefficients, intercepts, and R-squared values.
Clustering: K-Means clustering to segment employees based on lifestyle patterns.
Dashboard: Summary statistics and correlation matrix for an overall view.
Studies Navigation: Sidebar navigation to explore different studies.
Data Source
The data used in this project is sourced from the Kaggle: Time Management and Productivity Insights dataset.

Installation
Prerequisites
Python 3.10 or higher
Git (optional, for cloning the repository)
Clone the Repository
You can clone the repository using Git:

bash
Copy code
git clone https://github.com/unstableantimatter/streamlit_workforce-data.git
cd your-repo-name
Alternatively, you can download the repository as a ZIP file and extract it.

Set Up a Virtual Environment (Recommended)
It's recommended to use a virtual environment to manage dependencies.

bash
Copy code
python3 -m venv venv
Activate the virtual environment:

On Windows:

bash
Copy code
venv\Scripts\activate
On macOS/Linux:

bash
Copy code
source venv/bin/activate
Install Dependencies
Install the required Python packages using pip:

bash
Copy code
pip install -r requirements.txt
If you don't have a requirements.txt file, the main dependencies are:

bash
Copy code
pip install streamlit pandas numpy plotly scikit-learn
Usage
Run the Streamlit application:

bash
Copy code
streamlit run app.py
This will open the app in your default web browser at http://localhost:8501/.

Navigating the App
Use the sidebar to navigate between different studies and the dashboard.
Interact with the plots and charts to explore the data.
Read the insights and statistical analyses provided for each study.
Application Structure
app.py: The main application script containing all the code for the Streamlit app.
data/: Directory containing the dataset used by the app.
time_productivity_data.csv: The CSV file with productivity and lifestyle data.
requirements.txt: A file listing all the Python packages required to run the app.
README.md: Documentation for the project.
Dependencies
The application requires the following Python packages:

Streamlit: For creating the web application interface.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Plotly: For interactive visualizations.
Scikit-learn: For machine learning algorithms like Linear Regression and K-Means clustering.
Contributing
Contributions are welcome! If you'd like to improve this project, please fork the repository and create a pull request.

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.