import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
from etsi.failprint import analyze

st.set_page_config(page_title="FailPrint Dashboard", layout="wide")
st.title("📊 FailPrint – ML Failure Pattern Analyzer")

st.sidebar.header("Upload Prediction Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File with Input Features + y_true + y_pred", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    st.sidebar.header("Select Columns")
    feature_cols = st.sidebar.multiselect("Input Features", options=df.columns.tolist())
    y_true_col = st.sidebar.selectbox("True Labels (y_true)", options=df.columns.tolist())
    y_pred_col = st.sidebar.selectbox("Predicted Labels (y_pred)", options=df.columns.tolist())

    if feature_cols and y_true_col and y_pred_col:
        enable_filter = st.sidebar.checkbox("Enable Filtering", value=False)

        if enable_filter:
            filter_feature = st.sidebar.selectbox("Filter by Feature", options=feature_cols)
            selected_val = st.sidebar.selectbox(f"Select value from {filter_feature}", df[filter_feature].unique())
            df = df[df[filter_feature] == selected_val]
            st.markdown(f"**Filtering applied: {filter_feature} = {selected_val}**")

            if len(df) < 5:
                st.warning("⚠️ Too few rows after filtering. Try removing the filter for full analysis.")

        X = df[feature_cols]
        y_true = df[y_true_col]
        y_pred = df[y_pred_col]

        st.markdown("## 🧠 Failure Analysis Report")
        if len(df) >= 5:
            result = analyze(X, y_true, y_pred, output="markdown", cluster=True)
            st.markdown(result, unsafe_allow_html=True)

            markdown_report = result
            st.download_button(
                label="💾 Download Report as Markdown",
                data=markdown_report,
                file_name="failprint_report.md",
                mime="text/markdown"
            )
        else:
            st.warning("Not enough data to perform full analysis.")


        st.markdown("## 📉 Feature-wise Failure Distribution")
        misclassified = df[y_true != y_pred]

        if len(misclassified) >= 2:
            for feat in feature_cols:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.countplot(data=misclassified, x=feat, ax=ax)
                plt.title(f"Misclassification Distribution by {feat}")
                st.pyplot(fig)
        else:
            st.info("Not enough misclassified samples to show distribution.")


        st.markdown("## 🔍 Cluster Visualization (PCA)")
        if X.select_dtypes(include=['number']).shape[1] >= 2 and len(df) >= 5:
            try:
                pca = PCA(n_components=2)
                numeric_X = X.select_dtypes(include=['number'])
                components = pca.fit_transform(numeric_X)
                comp_df = pd.DataFrame(components, columns=["PC1", "PC2"])
                comp_df['Misclassified'] = (y_true != y_pred).astype(int)

                fig2 = px.scatter(
                comp_df, x="PC1", y="PC2",
                color=comp_df["Misclassified"].map({0: "Correct", 1: "Incorrect"}),
                title="PCA of Feature Space with Misclassification Highlight",
                )

                fig2.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
                st.plotly_chart(fig2)
            except:
                st.warning("PCA failed. Please ensure numeric features are selected.")
        else:
            st.info("PCA visualization skipped. Need at least 2 numeric features and 5 rows.")


        st.markdown("## 🔗 Feature Correlation Heatmap")
        if X.select_dtypes(include=['number']).shape[1] >= 2 and len(df) >= 5:
            corr = X.select_dtypes(include=['number']).corr()
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
            st.pyplot(fig3)
        else:
            st.info("Not enough numeric features or data for heatmap.")

        st.markdown("## ⚖️ Class Imbalance and Drift Analysis")
        if len(df) >= 2:
            fig4, ax4 = plt.subplots(1, 2, figsize=(14, 5))
            sns.countplot(x=y_true, ax=ax4[0])
            ax4[0].set_title("True Label Distribution")
            sns.countplot(x=y_pred, ax=ax4[1])
            ax4[1].set_title("Predicted Label Distribution")
            st.pyplot(fig4)
        else:
            st.info("Not enough data for class distribution.")

        with st.expander("ℹ️ Show Debugging Info"):
            st.write("Filtered Dataset Shape:", df.shape)
            st.write("Misclassified Samples Count:", sum(y_true != y_pred))
            st.dataframe(df[y_true != y_pred])



