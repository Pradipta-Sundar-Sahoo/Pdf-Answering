from transformers import pipeline

def load_qa_model():
    """Load a QA model and a summarization model from Hugging Face."""
    qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    summarizer = pipeline("summarization")
    return qa_model, summarizer

def get_answer(qa_model, summarizer, context, question):
    """Get answer to a question based on the given context using the QA model and summarize it."""
    result = qa_model(question=question, context=context, max_answer_len=100)
    detailed_answer = summarizer(result['answer'], max_length=130, min_length=0, do_sample=False)
    return detailed_answer[0]['summary_text']
