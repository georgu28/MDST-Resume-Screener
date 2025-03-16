from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
import io
from parser import read_pdf
from semantic import semantic_object
from knn_class import knn_pred

st.title("resume screener")

file = st.file_uploader("Upload PDF Resume", type="pdf")
if file:
    file_value = file.getvalue()
    pdf_viewer(file_value)

    with io.BytesIO(file_value) as f:
        text = read_pdf(f)
        st.write(text)
        
        # KNN Predictor
        model_class = knn_pred()
        predicted_resume_type = model_class.predict_pdf(f)

        st.write("KNN Model predicts that is is a",predicted_resume_type,"resume.")
        descriptions = ["full-stack","front-end","product-manager","java"]

        # Semantic Class Data
        semantic_class = semantic_object()
        semantic_class.clear_sentences()
        semantic_class.get_resume(f)
        for description in descriptions:
            semantic_class.get_description("pdfs/"+description+".pdf")
        
        st.write("Resume Semantic similarities:")

        similarities = []
        for i in range (1,5):
            similarities.append(round(semantic_class.get_similarities()[i].item(),3))
        
        resume_similarities = dict(zip(descriptions, similarities))
        st.write(resume_similarities,'\n')