from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

def load_qa_model():
    """Load the DistilBERT QA model and tokenizer from Hugging Face."""
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

def get_answer(qa_pipeline, context, question):
    """Get an answer to a question based on the given context using the QA pipeline."""
    result = qa_pipeline(question=question, context=context)
    return result['answer']
