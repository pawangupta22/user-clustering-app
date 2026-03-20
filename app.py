import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Advanced User Clustering", layout="wide")

# Title
st.title("Advanced Social Media User Clustering Dashboard")
st.markdown("Upload your dataset and perform **advanced clustering analysis** with insights.")

# File upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] < 2:
        st.error("Need at least 2 numeric columns")
    else:
        st.success("Numeric data detected")

        # Feature selection
        selected_features = st.multiselect(
            "Select features for clustering",
            numeric_df.columns,
            default=numeric_df.columns[:2]
        )

        if len(selected_features) < 2:
            st.warning("Select at least 2 features")
        else:
            data = numeric_df[selected_features]

            # Handle missing values
            data = data.fillna(data.mean())

            # Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)

            # ---------------- ELBOW METHOD ----------------
            st.subheader("Elbow Method (Find Optimal K)")

            inertia = []
            K_range = range(1, 11)

            for i in K_range:
                km = KMeans(n_clusters=i, random_state=42)
                km.fit(X_scaled)
                inertia.append(km.inertia_)

            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(K_range, inertia, marker='o')
            ax_elbow.set_xlabel("Number of Clusters (K)")
            ax_elbow.set_ylabel("Inertia")
            ax_elbow.set_title("Elbow Curve")
            st.pyplot(fig_elbow)

            # ---------------- CLUSTER SELECTION ----------------
            k = st.slider("Select number of clusters", 2, 10, 3)

            if st.button("Run Clustering"):
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)

                df['Cluster'] = clusters

                st.subheader("📌 Clustered Data")
                st.dataframe(df.head())

                # Cluster summary
                st.subheader("Cluster Summary")
                st.dataframe(df.groupby('Cluster')[selected_features].mean())

                # ---------------- VISUALIZATION ----------------
                st.subheader("Cluster Visualization")

                col1, col2 = st.columns(2)

                # Scatter plot
                with col1:
                    fig1, ax1 = plt.subplots()
                    scatter = ax1.scatter(
                        X_scaled[:, 0],
                        X_scaled[:, 1],
                        c=clusters,
                        cmap='viridis'
                    )
                    ax1.set_xlabel(selected_features[0])
                    ax1.set_ylabel(selected_features[1])
                    ax1.set_title("Cluster Scatter Plot")
                    st.pyplot(fig1)

                # Seaborn pairplot (advanced)
                with col2:
                    pairplot_df = df[selected_features + ['Cluster']]
                    fig2 = sns.pairplot(pairplot_df, hue='Cluster')
                    st.pyplot(fig2)

                # ---------------- DOWNLOAD ----------------
                st.subheader("⬇ Download Results")
                csv = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download Clustered Data",
                    data=csv,
                    file_name="clustered_data.csv",
                    mime="text/csv"
                )

                st.success("Clustering Completed Successfully!")
