import streamlit as st
import pandas as pd
from etsi.failprint import analyze

st.title("FailPrint - Simple Failure Analyzer")


uploaded_file = st.file_uploader("Upload your CSV file with model results", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(" Preview of uploaded data:")
    st.dataframe(data.head())

    # Ask user to select true and predicted label columns
    true_col = st.selectbox("Select the True Label Column", data.columns)
    pred_col = st.selectbox("Select the Predicted Label Column", data.columns)

    # Button to run analysis
    if st.button("Run Failure Analysis"):
        features = data.drop([true_col, pred_col], axis=1)
        true_labels = data[true_col]
        predicted_labels = data[pred_col]

        # Call failprint analyzer
        result = analyze(features, true_labels, predicted_labels, output="markdown", cluster=True)

        # Show the result
        st.markdown("### Analysis Report")
        st.markdown(result)
