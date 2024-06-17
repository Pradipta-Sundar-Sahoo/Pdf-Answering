import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file-like object."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    """Preprocess the extracted text for better QA results."""
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Remove stopwords and tokenize sentences into words
    stop_words = set(stopwords.words('english'))
    processed_text = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_sentence = ' '.join([word for word in words if word.lower() not in stop_words])
        processed_text.append(filtered_sentence)
    
    return ' '.join(processed_text)
