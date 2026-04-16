import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from model import process_data

st.set_page_config(page_title="User Clustering Dashboard", layout="wide")

st.title("📊 Social Media User Clustering Dashboard")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(data.head())

    # -----------------------------
    # Process Data
    # -----------------------------
    data = process_data(data)

    st.subheader("✅ Clustered Data")
    st.dataframe(data.head())

    # -----------------------------
    # Cluster Summary
    # -----------------------------
    st.subheader("📊 Cluster Summary")

    cluster_summary = data.groupby('Cluster')[[
        'Daily_Minutes_Spent',
        'Posts_Per_Day',
        'Likes_Per_Day',
        'Follows_Per_Day',
        'Engagement',
        'Activity_Score'
    ]].mean()

    st.dataframe(cluster_summary)

    # -----------------------------
    # Cluster Distribution
    # -----------------------------
    st.subheader("👥 Users per Cluster")

    fig1, ax1 = plt.subplots()
    data['Cluster'].value_counts().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

    # -----------------------------
    # User Type Distribution
    # -----------------------------
    st.subheader("🧠 User Types")

    fig2, ax2 = plt.subplots()
    data['User_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    st.pyplot(fig2)

    # -----------------------------
    # PCA Visualization
    # -----------------------------
    st.subheader("📍 Cluster Visualization (PCA)")

    features = [
        'Daily_Minutes_Spent',
        'Posts_Per_Day',
        'Likes_Per_Day',
        'Follows_Per_Day',
        'Engagement',
        'Activity_Score'
    ]

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data[features])

    data['PCA1'] = pca_data[:, 0]
    data['PCA2'] = pca_data[:, 1]

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, ax=ax3)
    st.pyplot(fig3)

    # -----------------------------
    # Top Users
    # -----------------------------
    st.subheader("🔥 Top 10 High Engagement Users")

    top_users = data.sort_values(by='Engagement', ascending=False).head(10)
    st.dataframe(top_users[['User_ID', 'Engagement', 'Cluster', 'User_Type']])

    # -----------------------------
    # Download Result
    # -----------------------------
    st.download_button(
        label="📥 Download Clustered Data",
        data=data.to_csv(index=False),
        file_name="clustered_users.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to start analysis.")
