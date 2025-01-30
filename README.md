# HR-Employee-Analysis-Dashboard
Project Title: Job Analysis Dashboard
![Screenshot 2024-08-11 120611](https://github.com/user-attachments/assets/7d61de78-b567-42c2-a36f-246711c63786)

Overview
The Job Analysis Dashboard is a Streamlit-based web application designed to perform various data analysis and text mining operations on a dataset related to job roles, departments, divisions, and responsibilities within an organization. Users can upload an Excel file containing job-related data, and the application provides various visualization and analysis features, including filtering options, text mining (like TF-IDF and word frequency analysis), and customized visualizations.
![Screenshot 2024-08-11 120646](https://github.com/user-attachments/assets/38ff030e-1177-4542-b466-1829408d78e7)


Features
File Upload
![Screenshot 2024-08-11 120707](https://github.com/user-attachments/assets/155357de-18db-43bd-a8b5-0e8122146e3d)

Users can upload an Excel file containing job-related data.
The uploaded file is read into a Pandas DataFrame, and the column names are displayed for verification.

![Screenshot 2024-08-11 120742](https://github.com/user-attachments/assets/19d15ba4-356d-4361-afe6-ce6c0f21e131)


Filtering Options

![Screenshot 2024-08-11 120806](https://github.com/user-attachments/assets/b2da14c4-4ba1-4d76-8698-083ffa3888bd)


Initial Filtering: Users can filter the dataset by selecting a specific column and value. This allows for a more focused analysis on subsets of the data.
Advanced Filtering: Additional filtering options are provided for categories like "Category," "Country," and "Job Title," allowing for more granular control over the data being analyzed.
Visualizations

![Screenshot 2024-08-11 120821](https://github.com/user-attachments/assets/0a18dd5f-3790-4f5c-b8e5-daa451a73331)

Job Titles Count: Visualizes the count of distinct job titles within departments and divisions. This feature can help in understanding the distribution of job roles across different organizational units.

![Screenshot 2024-08-11 120849](https://github.com/user-attachments/assets/dd15f4f4-3dd5-4c46-a480-bcfc7aec3097)

Job Titles with Hours >< 176: This visualization focuses on job titles where the total hours per month are either above or below 176. This can highlight potential overworking or underworking patterns.

![Screenshot 2024-08-11 120911](https://github.com/user-attachments/assets/58663479-e5b5-498b-a97b-391a0c805467)

Distinct Job Titles by Division/Department: Displays the distinct count of job titles by either division or department, providing insights into the diversity of roles within different organizational units.

![Screenshot 2024-08-11 120957](https://github.com/user-attachments/assets/074d64f4-74bc-4152-8eb6-44ccdbd792e2)

Department and Job Title Counts by Country: Visualizes the number of distinct departments and job titles within each country. This is useful for multinational organizations to understand the spread of roles across different locations.
Main Responsibility Analysis: Groups and counts main responsibilities by department and job title. It also allows filtering based on the count of responsibilities (e.g., <8 or >8).
Text Mining

![Screenshot 2024-08-11 121014](https://github.com/user-attachments/assets/071c3e58-da3c-49d1-be89-3be9b99d1710)

TF-IDF Analysis: The application performs a TF-IDF (Term Frequency-Inverse Document Frequency) analysis on the "Main Responsibility" column, with lemmatization provided by the SpaCy NLP library. This feature helps identify the most important terms within job responsibilities, which can be useful for understanding key focus areas in job roles.
Word Frequency Analysis: Analyzes the frequency of words in the "Main Responsibility" column, excluding common stop words. This feature highlights the most common terms used in job responsibilities.
Caching

The application uses Streamlit's new caching mechanisms (st.cache_resource) to optimize performance, especially when loading NLP models and processing large datasets.
Technical Details
Libraries Used:

![Screenshot 2024-08-11 121035](https://github.com/user-attachments/assets/34b6730d-00b4-484e-badc-5d5818df7a05)

Streamlit: For building the web application.
Pandas: For data manipulation and analysis.
Plotly Express & Graph Objects: For creating interactive visualizations.
Scikit-learn: For performing TF-IDF analysis.
SpaCy: For natural language processing, particularly lemmatization.
Regular Expressions (re): For custom text tokenization.
Data Requirements:

