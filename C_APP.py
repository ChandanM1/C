import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Streamlit page configuration
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 20px;
            background-color: #f4f4f4;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess dataset
@st.cache_data
def load_data(csv_path):
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not read the CSV file with any standard encoding")

    required_columns = ['Age', 'Income (INR)', 'Spending (1-100)', 'Gender']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Encode Gender column
    le = LabelEncoder()
    df['Gender_Encoded'] = le.fit_transform(df['Gender'])

    return df

# Perform clustering
@st.cache_data
def perform_clustering(df, method, num_clusters=None, eps=None, min_samples=None):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['Age', 'Income (INR)', 'Spending (1-100)']])

    if method == 'KMeans':
        model = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = model.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, df['Cluster'])

    elif method == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        df['Cluster'] = model.fit_predict(scaled_data)
        # Exclude noise (-1) from silhouette score
        valid_clusters = df['Cluster'] >= 0
        score = silhouette_score(scaled_data[valid_clusters], df['Cluster'][valid_clusters]) if valid_clusters.any() else -1

    elif method == 'Hierarchical':
        linkage_matrix = linkage(scaled_data, method='ward')
        df['Cluster'] = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')
        score = silhouette_score(scaled_data, df['Cluster'])

    return df, score

# Main app function
def main():
    # Display header
    st.markdown('<div class="main-header"><h1>Customer Segmentation System</h1></div>', unsafe_allow_html=True)

    # Sidebar inputs
    st.sidebar.header("🔧 Configuration")

    uploaded_file = st.sidebar.file_uploader("Upload Customer Dataset", type=['csv'])

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success("Dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading dataset: {e}")
            return

        # Select clustering method
        clustering_method = st.sidebar.selectbox("Select Clustering Method", ["KMeans", "DBSCAN", "Hierarchical"])

        # Parameters for clustering
        if clustering_method == 'KMeans':
            num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
            df, score = perform_clustering(df, method='KMeans', num_clusters=num_clusters)
        elif clustering_method == 'DBSCAN':
            eps = st.sidebar.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, step=0.1, value=0.5)
            min_samples = st.sidebar.slider("Minimum Samples", min_value=1, max_value=20, value=5)
            df, score = perform_clustering(df, method='DBSCAN', eps=eps, min_samples=min_samples)
        elif clustering_method == 'Hierarchical':
            num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
            df, score = perform_clustering(df, method='Hierarchical', num_clusters=num_clusters)

        st.sidebar.info(f"Silhouette Score: {score:.2f}")

        # Main visualization
        tab1, tab2, tab3 = st.tabs(["📊 Cluster Analysis", "📈 Visualizations", "📋 Full Dataset"])

        with tab1:
            st.markdown("### Cluster Analysis")
            cluster_summary = df.groupby('Cluster')[['Age', 'Income (INR)', 'Spending (1-100)']].mean()
            st.dataframe(cluster_summary)

        with tab2:
            st.markdown("### Visualizations")

            # Cluster Distribution
            cluster_counts = df['Cluster'].value_counts()
            plt.figure(figsize=(8, 6))
            plt.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title("Cluster Distribution")
            st.pyplot(plt)

            # Income vs Spending Scatter Plot
            plt.figure(figsize=(10, 6))
            for cluster in df['Cluster'].unique():
                cluster_data = df[df['Cluster'] == cluster]
                plt.scatter(cluster_data['Income (INR)'], cluster_data['Spending (1-100)'], label=f'Cluster {cluster}')
            plt.xlabel('Income (INR)')
            plt.ylabel('Spending (1-100)')
            plt.title('Income vs Spending by Cluster')
            plt.legend()
            st.pyplot(plt)

        with tab3:
            st.markdown("### Full Dataset")
            st.dataframe(df)

    else:
        st.sidebar.warning("Please upload a dataset to proceed.")

if __name__ == '__main__':
    main()
