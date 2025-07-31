from collections import defaultdict
import re
import sys
import pdfplumber
import logging
from typing import Dict, List, Tuple
from config import SECTION_TITLES, REGEX_PATTERNS

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_pdf(name: str) -> str:
    """
    Extract text from the first page of a PDF file.
    
    Args:
        name (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF cannot be read
    """
    try:
        with pdfplumber.open(name) as pdf:
            if not pdf.pages:
                logger.warning(f"PDF {name} has no pages")
                return ""
            # Extract text from all pages, not just the first one
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            return full_text
    except FileNotFoundError:
        logger.error(f"PDF file not found: {name}")
        raise
    except Exception as e:
        logger.error(f"Error reading PDF {name}: {str(e)}")
        raise


def matchSub(expr: str, text: str) -> Tuple[str, str]:
    """
    Find matches using regex pattern and return first match with cleaned text.
    
    Args:
        expr (str): Regular expression pattern
        text (str): Text to search in
        
    Returns:
        Tuple[str, str]: First match found and text with pattern removed
    """
    matches = re.findall(expr, text, re.IGNORECASE)
    cleaned = re.sub(expr, "", text, flags=re.IGNORECASE)
    # return the first match if it exists along with the cleaned text
    return matches[0] if matches else "", cleaned


def get_email(text: str) -> Tuple[str, str]:
    """
    Extract email address from resume text using regex pattern.
    
    Args:
        text (str): Resume text to search
        
    Returns:
        Tuple[str, str]: Email found and cleaned text
    """
    return matchSub(REGEX_PATTERNS['email'], text)


def get_phone(text: str) -> Tuple[str, str]:
    """
    Extract phone number from resume text.
    
    Args:
        text (str): Resume text to search
        
    Returns:
        Tuple[str, str]: Phone number found and cleaned text
    """
    return matchSub(REGEX_PATTERNS['phone'], text)


def get_linkedin(text: str) -> Tuple[str, str]:
    """
    Extract LinkedIn profile URL from resume text.
    
    Args:
        text (str): Resume text to search
        
    Returns:
        Tuple[str, str]: LinkedIn URL found and cleaned text
    """
    return matchSub(REGEX_PATTERNS['linkedin'], text)


def get_gpa(text: str) -> Tuple[str, str]:
    """
    Extract GPA information from resume text.
    
    Args:
        text (str): Resume text to search
        
    Returns:
        Tuple[str, str]: GPA found and cleaned text
    """
    return matchSub(REGEX_PATTERNS['gpa'], text)


def get_details(text: str) -> Tuple[Dict[str, str], str]:
    """
    Extract multiple personal details from resume text.
    
    Args:
        text (str): Resume text to parse
        
    Returns:
        Tuple[Dict[str, str], str]: Dictionary of extracted details and cleaned text
    """
    details = dict()
    details["gpa"], text = get_gpa(text)
    details["email"], text = get_email(text)
    details["phone"], text = get_phone(text)
    details["linkedin"], text = get_linkedin(text)
    return details, text


def section_title(line: str) -> str:
    """
    Identify if a line contains a section title.
    
    Args:
        line (str): Line of text to check
        
    Returns:
        str: Section title if found, empty string otherwise
    """
    words = line.lower().split()
    if len(words) > 2:
        return ""
    for word in words:
        if word in SECTION_TITLES:
            return word
    return ""


def sections(lines: List[str]) -> Dict[str, List[str]]:
    """
    Organize resume lines into sections based on section titles.
    
    Args:
        lines (List[str]): List of text lines from resume
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping section names to content
    """
    sectionData = defaultdict(list)

    section = ""
    for line in lines:
        if not len(line):
            continue
        res = section_title(line)
        if res:
            section = res
        elif len(section):
            sectionData[section].append(line)

    return sectionData


def get_sections(text: str) -> Dict[str, List[str]]:
    """
    Parse resume text into organized sections.
    
    Args:
        text (str): Full resume text
        
    Returns:
        Dict[str, List[str]]: Dictionary of sections and their content
    """
    return sections(text.split("\n"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("give path")
        exit(1)
    name = sys.argv[1]
    text = read_pdf(name)
    print(text)
    details, text = get_details(text)
    sections = get_sections(text)
    print(sections)
    print(details)