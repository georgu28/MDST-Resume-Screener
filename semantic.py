import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from parser import read_pdf
from parser import get_sections
from sentence_transformers import SentenceTransformer
import sys
import re
import numpy as np
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticMatcher:
    """
    A class for calculating semantic similarity between resumes and job descriptions
    using sentence transformers and embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic matcher with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.sentences = []
            logger.info(f"Initialized semantic matcher with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            raise
    
    def clear_sentences(self) -> None:
        """Clear the sentences list to prepare for new comparison."""
        self.sentences.clear()

    def get_resume_content(self, resume_path: str) -> None:
        """
        Extract and process resume content from PDF file.
        
        Args:
            resume_path (str): Path to the resume PDF file
            
        Raises:
            Exception: If PDF cannot be read or processed
        """
        try:
            resume_data = read_pdf(resume_path)
            
            # Extract relevant sections for resume analysis
            sections_to_extract = ["experience", "projects", "skills"]
            combined_content = ""
            
            for section in sections_to_extract:
                section_content = self._get_section_data(resume_data, section)
                combined_content += section_content + " "
            
            if combined_content.strip():
                self.sentences.append(self._transform_text(combined_content))
                logger.info(f"Successfully processed resume: {resume_path}")
            else:
                logger.warning(f"No relevant content found in resume: {resume_path}")
                
        except Exception as e:
            logger.error(f"Error processing resume {resume_path}: {e}")
            raise
    
    def get_job_description_content(self, description_path: str) -> None:
        """
        Extract and process job description content from PDF file.
        
        Args:
            description_path (str): Path to the job description PDF file
            
        Raises:
            Exception: If PDF cannot be read or processed
        """
        try:
            description_data = read_pdf(description_path)
            
            # Extract relevant sections for job description analysis
            sections_to_extract = ["responsibilities", "requirements", "description"]
            combined_content = ""
            
            for section in sections_to_extract:
                section_content = self._get_section_data(description_data, section)
                combined_content += section_content + " "
            
            if combined_content.strip():
                self.sentences.append(self._transform_text(combined_content))
                logger.info(f"Successfully processed job description: {description_path}")
            else:
                logger.warning(f"No relevant content found in job description: {description_path}")
                
        except Exception as e:
            logger.error(f"Error processing job description {description_path}: {e}")
            raise

    def _get_section_data(self, text: str, section: str) -> str:
        """
        Extract specific section data from text.
        
        Args:
            text (str): Full text content
            section (str): Section name to extract
            
        Returns:
            str: Joined content from the specified section
        """
        try:
            sections = get_sections(text)
            return ' '.join(sections.get(section, []))
        except Exception as e:
            logger.error(f"Error extracting section '{section}': {e}")
            return ""
    
    def calculate_similarities(self) -> Optional[np.ndarray]:
        """
        Calculate semantic similarities between stored sentences.
        
        Returns:
            Optional[np.ndarray]: Array of similarity scores, or None if calculation fails
        """
        if len(self.sentences) < 2:
            logger.warning("Need at least 2 sentences to calculate similarity")
            return None
            
        try:
            # Calculate embeddings by calling model.encode()
            embeddings = self.model.encode(self.sentences)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")

            # Calculate the embedding similarities
            similarities = self.model.similarity(embeddings, embeddings)
            return similarities[0]  # Return similarities for first sentence (resume)
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return None
        

    def _transform_text(self, text: str) -> str:
        """
        Clean and preprocess text for better semantic analysis.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase for consistency
        text = text.lower()

        # Remove URLs and links
        text = re.sub(r"http\S+", " ", text)
        
        # Remove non-ASCII characters
        text = re.sub(r"[^\x00-\x7f]", " ", text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def batch_compare_resumes(self, resume_paths: List[str], job_description_paths: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple resumes against multiple job descriptions.
        
        Args:
            resume_paths (List[str]): List of resume PDF paths
            job_description_paths (List[str]): List of job description PDF paths
            
        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary with similarity scores
        """
        results = {}
        
        for resume_path in resume_paths:
            resume_name = resume_path.split('/')[-1].replace('.pdf', '')
            self.clear_sentences()
            
            try:
                self.get_resume_content(resume_path)
                job_similarities = {}
                
                for job_path in job_description_paths:
                    job_name = job_path.split('/')[-1].replace('.pdf', '')
                    
                    # Add job description to comparison
                    self.get_job_description_content(job_path)
                    
                    # Calculate similarity
                    similarities = self.calculate_similarities()
                    if similarities is not None and len(similarities) > 1:
                        job_similarities[job_name] = round(similarities[1].item(), 3)
                    
                    # Remove job description for next iteration
                    if len(self.sentences) > 1:
                        self.sentences.pop()
                
                results[resume_name] = job_similarities
                
            except Exception as e:
                logger.error(f"Error processing resume {resume_name}: {e}")
                results[resume_name] = {}
        
        return results

    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python semantic.py <resume_path> <job_description_path>")
        exit(1)
    
    # Example usage for individual comparison
    resume_path = sys.argv[1]
    job_path = sys.argv[2]
    
    matcher = SemanticMatcher()
    matcher.get_resume_content(resume_path)
    matcher.get_job_description_content(job_path)
    
    similarities = matcher.calculate_similarities()
    if similarities is not None and len(similarities) > 1:
        similarity_score = round(similarities[1].item(), 3)
        print(f"Similarity between resume and job description: {similarity_score}")
    
    # Example batch processing
    resumes = ["pdfs/bryan-resume.pdf", "pdfs/john-resume.pdf", "pdfs/jakes-resume.pdf"]
    descriptions = ["pdfs/full-stack.pdf", "pdfs/front-end.pdf", "pdfs/product-manager.pdf", "pdfs/java.pdf"]

    batch_results = matcher.batch_compare_resumes(resumes, descriptions)
    
    print("\nBatch Comparison Results:")
    for resume_name, similarities in batch_results.items():
        print(f"{resume_name}: {similarities}")
        
    
    

