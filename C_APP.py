# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Streamlit app
st.title("Customer Segmentation: K-Means, DBSCAN, and Hierarchical Clustering")
st.write("This application allows customer segmentation using three clustering algorithms: **K-Means**, **DBSCAN**, and **Hierarchical Clustering**.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(data.head())

    # Select features for clustering
    features = st.multiselect(
        "Select Features for Clustering",
        options=data.columns.tolist(),
        default=data.columns[:2].tolist(),
    )
    
    if len(features) < 2:
        st.error("Please select at least two features for clustering.")
    else:
        # Extract selected features and scale them
        X = data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.write("### Scaled Features Preview:")
        st.write(pd.DataFrame(X_scaled, columns=features).head())

        # Select clustering algorithm
        algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN", "Hierarchical Clustering"])

        if algorithm == "K-Means":
            # K-Means Clustering
            st.write("### K-Means Clustering")
            n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

            # Fit K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            y_kmeans = kmeans.fit_predict(X_scaled)

            # Silhouette Score
            silhouette = silhouette_score(X_scaled, y_kmeans)
            st.write(f"Silhouette Score: {silhouette:.2f}")

            # Visualize clusters
            if len(features) == 2:
                plt.figure(figsize=(8, 5))
                plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap="viridis", marker="o")
                plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c="red", label="Centroids")
                plt.title("K-Means Clustering")
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.legend()
                st.pyplot(plt)

            # Add cluster labels to data
            data["Cluster"] = y_kmeans

        elif algorithm == "DBSCAN":
            # DBSCAN Clustering
            st.write("### DBSCAN Clustering")
            eps = st.slider("Select Epsilon (eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
            min_samples = st.slider("Select Minimum Samples", min_value=2, max_value=20, value=5)

            # Fit DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            y_dbscan = dbscan.fit_predict(X_scaled)

            # Number of clusters
            n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
            st.write(f"Number of Clusters: {n_clusters}")

            # Visualize clusters
            if len(features) == 2:
                plt.figure(figsize=(8, 5))
                plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap="viridis", marker="o")
                plt.title("DBSCAN Clustering")
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.grid(True)
                st.pyplot(plt)

            # Add cluster labels to data
            data["Cluster"] = y_dbscan

        elif algorithm == "Hierarchical Clustering":
            # Hierarchical Clustering
            st.write("### Hierarchical Clustering")
            plt.figure(figsize=(10, 7))
            dendrogram = sch.dendrogram(sch.linkage(X_scaled, method="ward"))
            st.pyplot(plt)

            n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

            # Fit Hierarchical Clustering
            hc = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
            y_hc = hc.fit_predict(X_scaled)

            # Visualize clusters
            if len(features) == 2:
                plt.figure(figsize=(8, 5))
                plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_hc, cmap="viridis", marker="o")
                plt.title("Hierarchical Clustering")
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.grid(True)
                st.pyplot(plt)

            # Add cluster labels to data
            data["Cluster"] = y_hc

        # Display clustered dataset
        st.write("### Clustered Data")
        st.write(data.head())

        # Display cluster statistics
        st.write("### Cluster Statistics")
        cluster_summary = data.groupby("Cluster").mean()
        st.write(cluster_summary)

        # Allow download of clustered dataset
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Clustered Data",
            data=csv,
            file_name="clustered_customers.csv",
            mime="text/csv",
        )
