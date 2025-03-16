import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from parser import read_pdf
from parser import get_sections
from sentence_transformers import SentenceTransformer
import sys
import re

class semantic_object:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = []
    
    def clear_sentences(self):
        self.sentences.clear()

    def get_resume(self,resume):
        resume_data = read_pdf(resume)
        resume_section_data = self.get_section_data(resume_data,"experience")
        resume_section_data = resume_section_data + self.get_section_data(resume_data,"projects")
        resume_section_data = resume_section_data + self.get_section_data(resume_data,"skills")
        self.sentences.append(self.transform_text(resume_section_data))

        # print(self.sentences)
    
    def get_description(self,description):
        description_data = read_pdf(description)
        description_section_data = self.get_section_data(description_data,"responsibilities")
        description_section_data = description_section_data + self.get_section_data(description_data,"requirements")
        self.sentences.append(self.transform_text(description_section_data))

    def get_section_data(self,text,section):
        sections = get_sections(text)
        string = ''.join(sections[section])
        return string
    
    def get_similarities(self):
        # Calculate embeddings by calling model.encode()
        embeddings = self.model.encode(self.sentences)
        # print(embeddings.shape)

        # Calculate the embedding similarities
        similarities = self.model.similarity(embeddings, embeddings)
        # print(similarities)
        return similarities[0]
        

    def transform_text(self,text):
        text = text.lower()

        # clean links
        text = re.sub(r"http\S+", " ", text)
        # remove all non ascii
        text = re.sub(r"[^\x00-\x7f]", " ", text)
        # remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        return text

    
if __name__ == "__main__":
    if len(sys.argv) < 0:
        print("give two paths: resume description")
        exit(1)
    
    resumes = ["bryan-resume","john-resume","jakes-resume"]
    descriptions = ["full-stack","front-end","product-manager","java"]

    semantic_class = semantic_object()
    for resume in resumes:
        semantic_class.clear_sentences()
        semantic_class.get_resume("pdfs/"+resume+".pdf")
        for description in descriptions:
            semantic_class.get_description("pdfs/"+description+".pdf")
        
        print(resume,"similarities")

        similarities = []
        for i in range (1,5):
            similarities.append(round(semantic_class.get_similarities()[i].item(),3))
        
        resume_similarities = dict(zip(descriptions, similarities))
        print(resume_similarities,'\n')
        
    
    

