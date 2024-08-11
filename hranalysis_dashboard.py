# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from sklearn.feature_extraction.text import TfidfVectorizer
# from collections import Counter
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# import plotly.graph_objects as go
# import spacy
# import re

# # Custom tokenizer to preprocess text data
# def custom_tokenizer(text):
#     tokens = text.split()
#     tokens = [token for token in tokens if not token.isdigit() and not re.fullmatch(r'\d+', token)]
#     filtered_text = " ".join(tokens)
#     return filtered_text

# @st.cache_resource
# def load_spacy_model(name):
#     return spacy.load(name)

# nlp = load_spacy_model("en_core_web_sm")

# def preprocess_text(text):
#     filtered_text = custom_tokenizer(text)
#     doc = nlp(filtered_text)
#     lemmatized_text = " ".join([token.lemma_ for token in doc])
#     return lemmatized_text

# # @st.cache
# def load_data():
#     # Load dataset
#     df = pd.read_csv('/content/drive/MyDrive/WA_Fn-UseC_-HR-Employee-Attrition.csv')
#     return df

# def main():
#     st.title('HR Employee Analysis Dashboard')

#     # Load data
#     df = load_data()

#     # Display column names
#     st.write("Column names in the dataset:", df.columns.tolist())

#     # Create tabs
#     tab1, tab2 = st.tabs(["Visualizations", "Text Mining"])

#     # Tab 1: Visualizations
#     with tab1:
#         filtered_df = df
#         column_names = df.columns.tolist()

#         # Initial filtering options
#         if st.sidebar.checkbox('Show Initial Filter Options'):
#             selected_column = st.sidebar.selectbox('Select a column to filter by', column_names)
#             unique_values = df[selected_column].unique()
#             selected_value = st.sidebar.selectbox(f'Select a value from {selected_column}', unique_values)
#             filtered_df = df[df[selected_column] == selected_value]

#         with st.expander("View Filtered Data"):
#             st.dataframe(filtered_df)

#         if st.checkbox('Show Job Roles Count'):
#             job_roles_df = filtered_df
#             job_roles = job_roles_df.groupby(['Department', 'JobRole'])['JobRole'].count().reset_index(name='Distinct Job Roles Count')
#             fig = px.bar(job_roles, x='JobRole', y='Distinct Job Roles Count', color='Department', title='Job Roles Count by Department')
#             st.plotly_chart(fig)
#             st.dataframe(job_roles)

#         if st.checkbox('Show Monthly Income Distribution'):
#             fig = px.histogram(filtered_df, x='MonthlyIncome', nbins=20, title='Monthly Income Distribution')
#             st.plotly_chart(fig)

#         if st.checkbox('Show Attrition by Department and Job Role'):
#             attrition_counts = filtered_df.groupby(['Department', 'JobRole'])['Attrition'].count().reset_index(name='Attrition Count')
#             fig = px.bar(attrition_counts, x='JobRole', y='Attrition Count', color='Department', title='Attrition Count by Job Role and Department')
#             st.plotly_chart(fig)
#             st.dataframe(attrition_counts)

#     # Tab 2: Text Mining
#     with tab2:
#         st.header("Text Mining")

#         st.write("TF-IDF Scores:")
#         custom_stop_words = list(ENGLISH_STOP_WORDS) + ['&', 'on', 'ibm', 'new', 'daily', 'monthly', '/', 'yearly']

#         if st.sidebar.checkbox('Show Advanced Filter Options'):
#             if 'Department' in column_names:
#                 departments = df['Department'].unique()
#                 selected_department = st.sidebar.selectbox('Filter by Department', ['All'] + list(departments))
#                 if selected_department != 'All':
#                     df = df[df['Department'] == selected_department]

#             if 'JobRole' in column_names:
#                 job_roles = df['JobRole'].unique()
#                 selected_job_role = st.sidebar.selectbox('Filter by JobRole', ['All'] + list(job_roles))
#                 if selected_job_role != 'All':
#                     df = df[df['JobRole'] == selected_job_role]

#         if 'JobRole' in df.columns:
#             job_roles = df['JobRole'].dropna().tolist()
#             if job_roles and st.checkbox('Perform TF-IDF Analysis on "JobRole"'):
#                 lemmatized_roles = [" ".join([token.lemma_ for token in nlp(custom_tokenizer(text))]) for text in job_roles]
#                 vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=10)
#                 tfidf_matrix = vectorizer.fit_transform(lemmatized_roles)
#                 tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
#                 st.write("TF-IDF Scores for 'JobRole'")
#                 st.dataframe(tfidf_df.head())
#                 styled_df = tfidf_df.head().style.background_gradient(cmap='viridis')
#                 st.dataframe(styled_df)

#                 top_keywords = tfidf_df.apply(lambda s: s.abs().nlargest(3).index.tolist(), axis=1)
#                 st.write("Top Keywords for Each Job Role (Sample)")
#                 st.dataframe(top_keywords.head())

#         if st.checkbox('Perform Word Frequency Analysis on "JobRole"'):
#             job_roles = df['JobRole'].dropna().tolist()
#             all_words = ' '.join(job_roles).lower().split()
#             filtered_words = [word for word in all_words if word not in custom_stop_words]
#             word_counts = Counter(filtered_words)
#             most_common_words = word_counts.most_common(10)
#             words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
#             st.write("Most Frequent Words in 'JobRole' (excluding common stop words)")
#             st.dataframe(words_df)
#             fig = px.bar(words_df, x='Word', y='Frequency', title="Top 10 Most Frequent Words (excluding common stop words)")
#             st.plotly_chart(fig)

# if __name__ == "__main__":
#     main()





import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
import re

# Custom tokenizer to preprocess text data
def custom_tokenizer(text):
    tokens = text.split()
    tokens = [token for token in tokens if not token.isdigit() and not re.fullmatch(r'\d+', token)]
    filtered_text = " ".join(tokens)
    return filtered_text

@st.cache_resource
def load_spacy_model(name):
    return spacy.load(name)

nlp = load_spacy_model("en_core_web_sm")

def preprocess_text(text):
    filtered_text = custom_tokenizer(text)
    doc = nlp(filtered_text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv('/content/drive/MyDrive/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    return df

def main():
    st.title('HR Employee Analysis Dashboard')

    # Load data
    df = load_data()

    # Display column names
    st.write("Column names in the dataset:", df.columns.tolist())

    # Create tabs
    tab1, tab2 = st.tabs(["Visualizations", "Text Mining"])

    # Tab 1: Visualizations
    with tab1:
        filtered_df = df
        column_names = df.columns.tolist()

        # Initial filtering options
        if st.sidebar.checkbox('Show Initial Filter Options'):
            selected_column = st.sidebar.selectbox('Select a column to filter by', column_names)
            unique_values = df[selected_column].unique()
            selected_value = st.sidebar.selectbox(f'Select a value from {selected_column}', unique_values)
            filtered_df = df[df[selected_column] == selected_value]

        with st.expander("View Filtered Data"):
            st.dataframe(filtered_df)

        if st.checkbox('Show Job Roles Count'):
            job_roles_df = filtered_df
            job_roles = job_roles_df.groupby(['Department', 'JobRole'])['JobRole'].count().reset_index(name='Distinct Job Roles Count')
            fig = px.bar(job_roles, x='JobRole', y='Distinct Job Roles Count', color='Department', title='Job Roles Count by Department')
            st.plotly_chart(fig)
            st.dataframe(job_roles)

        if st.checkbox('Show Monthly Income Distribution'):
            fig = px.histogram(filtered_df, x='MonthlyIncome', nbins=20, title='Monthly Income Distribution')
            st.plotly_chart(fig)

        if st.checkbox('Show Attrition by Department and Job Role'):
            attrition_counts = filtered_df.groupby(['Department', 'JobRole'])['Attrition'].count().reset_index(name='Attrition Count')
            fig = px.bar(attrition_counts, x='JobRole', y='Attrition Count', color='Department', title='Attrition Count by Job Role and Department')
            st.plotly_chart(fig)
            st.dataframe(attrition_counts)

    # Tab 2: Text Mining
    with tab2:
        st.header("Text Mining")

        st.write("TF-IDF Scores:")
        custom_stop_words = list(ENGLISH_STOP_WORDS) + ['&', 'on', 'ibm', 'new', 'daily', 'monthly', '/', 'yearly']

        if st.sidebar.checkbox('Show Advanced Filter Options'):
            if 'Department' in column_names:
                departments = df['Department'].unique()
                selected_department = st.sidebar.selectbox('Filter by Department', ['All'] + list(departments))
                if selected_department != 'All':
                    df = df[df['Department'] == selected_department]

            if 'JobRole' in column_names:
                job_roles = df['JobRole'].unique()
                selected_job_role = st.sidebar.selectbox('Filter by JobRole', ['All'] + list(job_roles))
                if selected_job_role != 'All':
                    df = df[df['JobRole'] == selected_job_role]

        if 'JobRole' in df.columns:
            job_roles = df['JobRole'].dropna().tolist()
            if job_roles and st.checkbox('Perform TF-IDF Analysis on "JobRole"'):
                lemmatized_roles = [" ".join([token.lemma_ for token in nlp(custom_tokenizer(text))]) for text in job_roles]
                vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=10)
                tfidf_matrix = vectorizer.fit_transform(lemmatized_roles)
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
                st.write("TF-IDF Scores for 'JobRole'")
                st.dataframe(tfidf_df.head())
                styled_df = tfidf_df.head().style.background_gradient(cmap='viridis')
                st.dataframe(styled_df)

                top_keywords = tfidf_df.apply(lambda s: s.abs().nlargest(3).index.tolist(), axis=1)
                st.write("Top Keywords for Each Job Role (Sample)")
                st.dataframe(top_keywords.head())

        if st.checkbox('Perform Word Frequency Analysis on "JobRole"'):
            job_roles = df['JobRole'].dropna().tolist()
            all_words = ' '.join(job_roles).lower().split()
            filtered_words = [word for word in all_words if word not in custom_stop_words]
            word_counts = Counter(filtered_words)
            most_common_words = word_counts.most_common(10)
            words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
            st.write("Most Frequent Words in 'JobRole' (excluding common stop words)")
            st.dataframe(words_df)
            fig = px.bar(words_df, x='Word', y='Frequency', title="Top 10 Most Frequent Words (excluding common stop words)")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    st.subheader('Made By Deepak Kumar')