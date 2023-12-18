import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import pdfplumber
import streamlit as st


class QuestionAnswerBot:
    def __init__(self, context, chunk_size=200):
        self.chunk_size = chunk_size
        self.context_chunks = self.chunk_context(context)
        self.vectorizer = TfidfVectorizer()
        self.context_vectors = self.vectorizer.fit_transform(self.context_chunks)

    def chunk_context(self, context):
        sentences = context.split('.')
        chunks = [' '.join(sentences[i:i + self.chunk_size]) for i in range(0, len(sentences), self.chunk_size)]
        return chunks

    def find_relevant_chunk(self, question):
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.context_vectors)
        most_relevant_chunk_index = np.argmax(similarities)
        return self.context_chunks[most_relevant_chunk_index]

    def generate_answer(self, question):
        relevant_chunk = self.find_relevant_chunk(question)

        # Configure your Google Generative AI model
        genai.configure(api_key="AIzaSyDLQehxg9kSK5MqCI1N0GiDrcT9Re-XV-c")
        generation_config = {
            "temperature": 0.05,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 20480,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]

        model = genai.GenerativeModel(model_name="gemini-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        prompt_parts = [relevant_chunk, question]
        response = model.generate_content(prompt_parts)
        # Check if the response is satisfactory
        if self.is_response_satisfactory(response.text):
            return response.text
        else:
            return "I'm sorry, I don't have enough information to answer that question accurately."

    def is_response_satisfactory(self, response):
        # Implement logic to determine if the response is satisfactory
        # For example, check if the response is too short, or if it contains phrases like "I don't know"
        if len(response) < 30 or "I don't know" in response:
            return False
        return True

def read_pdf(pdf_path):
    pdf_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text() or ""
    return pdf_content

def handle_question_input():
    question = st.session_state.get('question', '')
    if question:
        st.session_state['questions_asked'].append(question)
        generate_and_display_answer(question)
        st.session_state['question'] = ''  # Clear the input box after processing

# Google Generative AI configuration
genai.configure(api_key="AIzaSyDLQehxg9kSK5MqCI1N0GiDrcT9Re-XV-c")

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

genai_model = genai.GenerativeModel(model_name="gemini-pro",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)

def generate_and_display_answer(question):
    answer = qa_bot.generate_answer(question)

    # If the answer is not satisfactory, ask the generative AI model
    if "I'm sorry, I don't have enough information to answer that question accurately." in answer:
        # Append the current question to the history for context
        history = st.session_state['history'] + [{"role": "user", "parts": question}]
        
        # Check and manage history length for the Google Generative AI model
        total_words = sum(len(entry["parts"].split()) for entry in history)
        if total_words > 30000:
            history = [{"role": "user", "parts": question}]

        convo = genai_model.start_chat(history=history)
        convo.send_message(question)
        answer = convo.last.text

        # Update the history with the model's response
        st.session_state['history'].append({"role": "model", "parts": answer})
    else:
        # Update the history with both question and answer
        st.session_state['history'].append({"role": "user", "parts": question})
        st.session_state['history'].append({"role": "model", "parts": answer})

    # Display the answer
    st.write(f"Answer: {answer}")


pdf_content = read_pdf('1678899842229.pdf')
qa_bot = QuestionAnswerBot(pdf_content)

def main():
    st.title("Bench Sales Recruiter Training Bot")
    st.write("PDF content has been loaded. You can now ask questions based on the PDF:")
    

    custom_css = """
    <style>
    button[kind="primary"].fixed-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #007bff;
        color: #fff;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }

    button[kind="primary"].fixed-button:hover {
        background-color: #0056b3;
    }

    </style>
    """



    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Create Streamlit app with sidebar
    st.sidebar.title("Sidebar")

    # Define buttons
    clear_input_button = st.sidebar.button('Clear Input', type="primary")
    clear_output_button = st.sidebar.button('Clear Output', type="primary")
    clear_both_button = st.sidebar.button('Clear Both', type="primary")

    # Define actions when buttons are clicked
    if clear_input_button:
        st.session_state['question'] = ''
    if clear_output_button:
        st.session_state['history'] = []
    if clear_both_button:
        st.session_state['question'] = ''
        st.session_state['history'] = []


    # Suggested prompts
    suggested_prompts = ["What is the main topic of the PDF?", "Summarize the first chapter.", "List the key points discussed."]

    # Initialize conversation history in session state if not already present
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Handle suggested prompt button click
    for prompt in suggested_prompts:
        if st.button(f"Suggested: {prompt}"):
            st.session_state['question'] = prompt
            generate_and_display_answer(prompt)
            break

    # Text input for the question
    question = st.text_input("Ask a question:", value=st.session_state.get('question', ''), key="question")

    # Check if the Ask button was clicked
    if st.button("Ask", key="ask_button"):
        if question:
            generate_and_display_answer(question)
            st.session_state['question'] = ''  # Clear the input box after processing


if __name__ == "__main__":
    main()