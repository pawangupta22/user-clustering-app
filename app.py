import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page settings
st.set_page_config(page_title="User Clustering App", layout="centered")

# Title
st.title("📊 Social Media User Clustering")
st.write("Upload your dataset and get user clusters instantly 🚀")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] < 2:
        st.error("❌ Need at least 2 numeric columns for clustering")
    else:
        st.success("✅ Data ready for clustering")

        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        # Choose clusters
        k = st.slider("Select number of clusters", 2, 10, 3)

        # Run clustering
        if st.button("Run Clustering"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            df['Cluster'] = clusters

            st.subheader("📊 Clustered Data")
            st.write(df.head())

            # Cluster summary
            st.subheader("📈 Cluster Summary")
            st.write(df.groupby('Cluster').mean())

            # Graph
            st.subheader("📉 Visualization")
            fig, ax = plt.subplots()
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            st.pyplot(fig)

            st.success("🎉 Clustering Completed Successfully")
