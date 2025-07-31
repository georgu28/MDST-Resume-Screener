from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
import io
import tempfile
import os
from parser import read_pdf
from semantic import SemanticMatcher
from knn_class import ResumeClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MDST Resume Screener",
    page_icon="📄",
    layout="wide"
)

st.title("📄 MDST Resume Screener")
st.markdown("Upload a resume PDF to analyze its content and get job category predictions")

# File uploader
file = st.file_uploader("Upload PDF Resume", type="pdf")

if file:
    try:
        # Display PDF
        file_value = file.getvalue()
        st.subheader("Resume Preview")
        pdf_viewer(file_value)

        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_value)
            tmp_file_path = tmp_file.name

        # Extract text from PDF
        st.subheader("Extracted Text")
        text = read_pdf(tmp_file_path)
        
        if text.strip():
            with st.expander("View extracted text"):
                st.text(text[:1000] + "..." if len(text) > 1000 else text)
            
            # KNN Classification
            st.subheader("🤖 Job Category Prediction")
            
            with st.spinner("Analyzing resume with KNN classifier..."):
                try:
                    classifier = ResumeClassifier()
                    predicted_category = classifier.predict_pdf(tmp_file_path)
                    
                    st.success(f"**Predicted Category:** {predicted_category}")
                    
                    # Get prediction probabilities
                    probabilities = classifier.get_prediction_probabilities(tmp_file_path)
                    
                    st.subheader("📊 Category Confidence Scores")
                    for category, prob in list(probabilities.items())[:5]:  # Top 5
                        st.write(f"**{category}:** {prob:.3f}")
                        st.progress(prob)
                    
                except Exception as e:
                    st.error(f"Error in KNN prediction: {str(e)}")
                    logger.error(f"KNN prediction error: {e}")

            # Semantic Similarity Analysis
            st.subheader("🔍 Semantic Similarity Analysis")
            
            job_descriptions = {
                "Full Stack Developer": "pdfs/full-stack.pdf",
                "Front End Developer": "pdfs/front-end.pdf", 
                "Product Manager": "pdfs/product-manager.pdf",
                "Java Developer": "pdfs/java.pdf"
            }
            
            with st.spinner("Calculating semantic similarities..."):
                try:
                    matcher = SemanticMatcher()
                    
                    # Process resume
                    matcher.get_resume_content(tmp_file_path)
                    
                    similarities = {}
                    for job_title, job_path in job_descriptions.items():
                        if os.path.exists(job_path):
                            # Add job description
                            matcher.get_job_description_content(job_path)
                            
                            # Calculate similarity
                            sim_scores = matcher.calculate_similarities()
                            if sim_scores is not None and len(sim_scores) > 1:
                                similarities[job_title] = round(sim_scores[1].item(), 3)
                            
                            # Remove job description for next iteration
                            if len(matcher.sentences) > 1:
                                matcher.sentences.pop()
                    
                    if similarities:
                        st.subheader("📈 Job Match Scores")
                        
                        # Sort by similarity score
                        sorted_similarities = dict(sorted(similarities.items(), 
                                                         key=lambda x: x[1], reverse=True))
                        
                        for job_title, score in sorted_similarities.items():
                            st.write(f"**{job_title}:** {score}")
                            st.progress(min(score, 1.0))  # Cap at 1.0 for progress bar
                        
                        # Recommend best match
                        best_match = max(similarities, key=similarities.get)
                        best_score = similarities[best_match]
                        st.success(f"🎯 **Best Match:** {best_match} (Score: {best_score})")
                    else:
                        st.warning("No similarities could be calculated")
                        
                except Exception as e:
                    st.error(f"Error in semantic analysis: {str(e)}")
                    logger.error(f"Semantic analysis error: {e}")
        else:
            st.error("No text could be extracted from the PDF")
            
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This resume screener uses:
    
    **🤖 KNN Classification**
    - Trained on resume dataset
    - Predicts job categories
    - Shows confidence scores
    
    **🔍 Semantic Analysis** 
    - Uses sentence transformers
    - Compares resume to job descriptions
    - Finds best job matches
    
    **📊 Features**
    - PDF text extraction
    - Multi-model analysis
    - Interactive results
    """)
    
    st.header("Supported Categories")
    st.markdown("""
    - Full Stack Developer
    - Front End Developer  
    - Product Manager
    - Java Developer
    - And more...
    """)