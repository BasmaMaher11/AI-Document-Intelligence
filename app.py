import streamlit as st
import tempfile
import os
import html
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)

from langchain_core.prompts import PromptTemplate


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CUSTOM GUI / CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(124, 58, 237, 0.12), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(168, 85, 247, 0.10), transparent 30%),
        #08060D;
    color: #F5F3FF;
}

.main {
    background: #08060D;
}

[data-testid="stDecoration"] {
    display: none;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* HERO */

.hero {
    background:
        radial-gradient(circle at 85% 20%, rgba(168, 85, 247, 0.28), transparent 34%),
        radial-gradient(circle at 10% 90%, rgba(124, 58, 237, 0.16), transparent 34%),
        linear-gradient(135deg, #171022, #0C0812);
    border: 1px solid rgba(168, 85, 247, 0.30);
    border-radius: 24px;
    padding: 42px 46px;
    margin-bottom: 30px;
    box-shadow: 0 0 45px rgba(139, 92, 246, 0.10);
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(192, 132, 252, 0.30);
    color: #D8B4FE;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 15px;
}

.hero-title {
    font-size: 43px;
    font-weight: 800;
    line-height: 1.15;
    margin: 0;
    color: #FFFFFF;
}

.hero-gradient {
    background: linear-gradient(90deg, #FFFFFF, #E9D5FF, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #AAA0B5;
    font-size: 16px;
    line-height: 1.7;
    max-width: 760px;
    margin: 12px 0 0 0;
}


/* SECTION TITLES */

.section-title {
    color: #F5F3FF;
    font-size: 24px;
    font-weight: 750;
    margin: 22px 0 5px 0;
}

.section-subtitle {
    color: #8F849A;
    font-size: 14px;
    margin-bottom: 16px;
}


/* CARDS */

.glass-card {
    background: linear-gradient(145deg, #191024, #100A16);
    border: 1px solid rgba(168, 85, 247, 0.20);
    border-radius: 17px;
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
}

.card-title {
    color: #E9D5FF;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 6px;
}

.card-text {
    color: #A79AAF;
    font-size: 14px;
    line-height: 1.6;
}


/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: linear-gradient(145deg, #160D20, #0E0914);
    border: 1px dashed #8B5CF6;
    border-radius: 17px;
    padding: 10px;
    color: #FFFFFF !important;
    box-shadow: 0 0 28px rgba(139, 92, 246, 0.08);
}

[data-testid="stFileUploaderDropzone"] {
    background: #120B19 !important;
    border-radius: 12px;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #21152E !important;
    color: #FFFFFF !important;
    border: 1px solid #8B5CF6 !important;
    border-radius: 9px !important;
}

[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] small *,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] section,
[data-testid="stFileUploaderDropzone"] section * {
    color: #FFFFFF !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #352047 !important;
    border-color: #C084FC !important;
}


/* TEXT AREA */

textarea,
input {
    background-color: #120B19 !important;
    color: #F5F3FF !important;
    border: 1px solid #382747 !important;
    border-radius: 12px !important;
}

textarea::placeholder,
input::placeholder {
    color: #74677E !important;
}

textarea:focus,
input:focus {
    border: 1px solid #A855F7 !important;
    box-shadow: 0 0 0 1px #A855F7,
                0 0 16px rgba(168, 85, 247, 0.16) !important;
}


/* BUTTONS */

.stButton > button {
    width: 100%;
    min-height: 44px;
    background: linear-gradient(135deg, #6D28D9, #A855F7);
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 15px;
    font-weight: 700;
    box-shadow: 0 7px 22px rgba(139, 92, 246, 0.20);
    transition: all 0.2s ease;
}

.stButton > button p {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(168, 85, 247, 0.35);
    background: linear-gradient(135deg, #7C3AED, #C084FC);
}


/* ANSWER BOX */

.answer-box {
    background: linear-gradient(145deg, #1B1027, #100A17);
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-left: 4px solid #A855F7;
    border-radius: 15px;
    padding: 22px;
    margin-top: 10px;
    color: #F3E8FF;
    font-size: 15px;
    line-height: 1.8;
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.06);
}

.answer-box p {
    color: #F3E8FF !important;
}


/* SOURCE BOX */

.source-box {
    background: #120B19;
    border: 1px solid #30213D;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 7px;
    color: #C4B5FD;
    font-size: 14px;
}


/* SUMMARY BOX */

.summary-box {
    background: linear-gradient(145deg, #1B1027, #100A17);
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-left: 4px solid #A855F7;
    border-radius: 15px;
    padding: 23px;
    margin-top: 10px;
    color: #EDE9FE;
    font-size: 15px;
    line-height: 1.85;
    box-shadow: 0 0 28px rgba(168, 85, 247, 0.06);
}


/* DOCUMENT INFO CARDS */

.info-card {
    background: linear-gradient(145deg, #160D20, #0E0915);
    border: 1px solid rgba(139, 92, 246, 0.20);
    border-radius: 14px;
    padding: 15px;
    text-align: center;
    margin-bottom: 16px;
}

.info-number {
    color: #C084FC;
    font-size: 24px;
    font-weight: 800;
}

.info-label {
    color: #8F849A;
    font-size: 11px;
    letter-spacing: 0.7px;
    margin-top: 2px;
}


/* ALERTS */

[data-testid="stAlert"] {
    background: #120C18;
    border-radius: 12px;
    border: 1px solid rgba(139, 92, 246, 0.25);
}


/* DIVIDER */

hr {
    border: none;
    border-top: 1px solid #251A30;
    margin: 30px 0;
}


/* SPINNER */

.stSpinner > div {
    border-top-color: #A855F7 !important;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #62576C;
    font-size: 12px;
    margin-top: 42px;
    padding-top: 18px;
    border-top: 1px solid #21172A;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-badge">✦ AI POWERED DOCUMENT ASSISTANT</div>
    <h1 class="hero-title">
        Smart <span class="hero-gradient">Document Intelligence</span>
    </h1>
    <p class="hero-subtitle">
        Upload your PDF, ask questions, and generate intelligent
        summaries using AI — all in one place.
    </p>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 4. LOAD MISTRAL MODEL
# ============================================================

@st.cache_resource
def load_llm():

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    return tokenizer, model


tokenizer, model = load_llm()


# ============================================================
# 5. TEXT GENERATION
# ============================================================

def generate_text(prompt, max_new_tokens=150):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# 6. EMBEDDING MODEL
# ============================================================

embedding_model_name = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# 7. CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectordb = FAISS.from_documents(chunks, embedding)

    return vectordb, chunks


# ============================================================
# 8. QA OUTPUT PARSER
# ============================================================

response_schemas = [
    ResponseSchema(
        name="answer",
        description=(
            "The answer to the user's question "
            "based only on the provided document context."
        )
    )
]

output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)

format_instructions = output_parser.get_format_instructions()


# ============================================================
# 9. QA PROMPT
# ============================================================

prompt_template = """
You are a helpful document assistant.

Answer the question using ONLY the information
provided in the context.

If the answer is not available in the context,
say:

"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

{format_instructions}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    partial_variables={
        "format_instructions": format_instructions
    }
)


# ============================================================
# 10. ASK QUESTION
# ============================================================

def ask_question(query, vectordb):

    docs = vectordb.similarity_search(query, k=3)

    context = "\n\n".join(
        [
            (
                f"[Page {doc.metadata.get('page', 'Unknown')}]"
                f"\n{doc.page_content}"
            )
            for doc in docs
        ]
    )

    formatted_prompt = prompt.format(
        context=context,
        question=query
    )

    result = generate_text(
        formatted_prompt,
        max_new_tokens=150
    )

    try:
        parsed_result = output_parser.parse(result)
    except Exception:
        parsed_result = {"answer": result}

    sources = sorted(
        set(
            doc.metadata.get("page", 0) + 1
            for doc in docs
            if doc.metadata.get("page") is not None
        )
    )

    return parsed_result, sources


# ============================================================
# 11. DOCUMENT SUMMARY
# ============================================================

def summarize_document(chunks):

    try:
        chunk_size = 4
        partial_summaries = []

        for i in range(0, len(chunks), chunk_size):

            batch = chunks[i:i + chunk_size]

            context = "\n\n".join(
                [doc.page_content for doc in batch]
            )

            summary_prompt = f"""
You are a helpful document summarization assistant.

Summarize the following document content clearly and concisely.

Focus on:
- Main ideas
- Important information
- Key points
- Important facts or conclusions

Do NOT return JSON.
Do NOT use markdown code blocks.
Return only the summary in plain text.

Document content:
{context}

Summary:
"""

            summary = generate_text(
                summary_prompt,
                max_new_tokens=200
            )

            if summary.strip():
                partial_summaries.append(summary.strip())

        if not partial_summaries:
            return None

        if len(partial_summaries) == 1:
            return partial_summaries[0]

        combined = "\n\n".join(partial_summaries)

        final_prompt = f"""
You are a document summarization assistant.

Below are summaries of different parts of the same document.

Combine them into ONE coherent final summary.

Remove repeated information.
Keep the most important points.
Do not invent information.
Do NOT return JSON.
Do NOT use code blocks.

Return only the final summary in plain text.

Partial summaries:
{combined}

Final summary:
"""

        final_summary = generate_text(
            final_prompt,
            max_new_tokens=300
        )

        return final_summary.strip()

    except Exception as e:
        st.error(f"Error while generating summary: {e}")
        return None


# ============================================================
# 12. INITIALIZE SESSION STATE
# ============================================================

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# Used to clear the input safely before the next rerun.
def ask_another_question():
    st.session_state.question_input = ""


# ============================================================
# 13. PDF UPLOAD
# ============================================================

st.markdown(
    """
<div class="section-title">📄 Upload Your Document</div>
<div class="section-subtitle">
    Upload a PDF document to start analyzing it with AI.
</div>
""",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Drop your PDF here or browse files",
    type=["pdf"],
    label_visibility="collapsed"
)


# ============================================================
# 14. PROCESS PDF AND DISPLAY APP
# ============================================================

if uploaded_file is not None:

    # Process only when a different file is uploaded.
    if st.session_state.document_name != uploaded_file.name:

        with st.spinner("Processing your PDF..."):

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                vectordb, chunks = create_vector_database(temp_path)

                total_pages = len(
                    set(
                        doc.metadata.get("page", 0)
                        for doc in chunks
                    )
                )

                st.session_state.vectordb = vectordb
                st.session_state.chunks = chunks
                st.session_state.document_name = uploaded_file.name
                st.session_state.total_pages = total_pages

                # Reset history for the new document.
                st.session_state.qa_history = []

                # The question widget has not been created yet.
                st.session_state.question_input = ""

            except Exception as e:
                st.error("An error occurred while processing the PDF.")
                st.exception(e)

            finally:
                if temp_path is not None and os.path.exists(temp_path):
                    os.remove(temp_path)

    if st.session_state.vectordb is None:
        st.stop()

    # ========================================================
    # DOCUMENT INFORMATION
    # ========================================================

    st.markdown(
        f"""
<div class="glass-card">
    <div class="card-title">📄 Current Document</div>
    <div class="card-text">
        {html.escape(st.session_state.document_name or "")}
    </div>
    <div class="card-text">
        Total pages: {st.session_state.total_pages}
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    # ========================================================
    # ASK AI ABOUT YOUR DOCUMENT (ONLY ONCE)
    # ========================================================

    st.markdown(
        """
<div class="section-title">
    💬 Ask AI About Your Document
</div>
<div class="section-subtitle">
    Ask questions and get answers based only on the
    information available in your PDF.
</div>
""",
        unsafe_allow_html=True
    )

    question = st.text_area(
        "Your question",
        placeholder="Example: What are the main objectives of this document?",
        height=110,
        label_visibility="collapsed",
        key="question_input"
    )

    col1, col2 = st.columns(2)

    with col1:
        ask_clicked = st.button(
            "🔍  Ask Question",
            key="ask_question_button",
            use_container_width=True
        )

    with col2:
        st.button(
            "➕  Ask Another Question",
            key="ask_another_question_button",
            on_click=ask_another_question,
            use_container_width=True
        )

    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if ask_clicked:

        if not question.strip():
            st.warning("Please enter a question first.")

        else:
            with st.spinner("Searching the document and generating an answer..."):
                try:
                    answer, sources = ask_question(
                        question,
                        st.session_state.vectordb
                    )

                    st.session_state.qa_history.append({
                        "question": question.strip(),
                        "answer": answer.get(
                            "answer",
                            "No answer was generated."
                        ),
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # ========================================================
    # DISPLAY QUESTION AND ANSWER HISTORY
    # ========================================================

    if st.session_state.qa_history:

        st.divider()

        for index, item in enumerate(st.session_state.qa_history, start=1):

            st.markdown(
                f"""
<div class="section-title">
    🧠 AI Answer #{index}
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown("**Your question:**")
            st.write(item["question"])

            st.markdown(
                f"""
<div class="answer-box">
    {html.escape(str(item["answer"]))}
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown(
                """
<div class="section-title">📚 Sources</div>
<div class="section-subtitle">
    Relevant pages used to generate the answer.
</div>
""",
                unsafe_allow_html=True
            )

            if item["sources"]:
                for page in item["sources"]:
                    st.markdown(
                        f"""
<div class="source-box">
    📄 &nbsp; Page {page}
</div>
""",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No source pages were identified.")

            st.divider()

        st.caption("Your previous answers will remain visible above.")

    # ========================================================
    # DOCUMENT SUMMARY
    # ========================================================

    st.markdown(
        """
<div class="section-title">📑 Document Summary</div>
<div class="section-subtitle">
    Generate a concise overview of the most important
    information in your document.
</div>
""",
        unsafe_allow_html=True
    )

    if st.button(
        "✨  Generate Summary",
        key="generate_summary_button",
        use_container_width=True
    ):

        with st.spinner("AI is generating your document summary..."):

            summary = summarize_document(st.session_state.chunks)

            if summary:
                st.markdown(
                    '<div class="section-title">📝 Summary</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
<div class="summary-box">
    {html.escape(summary)}
</div>
""",
                    unsafe_allow_html=True
                )
            else:
                st.warning("No summary could be generated.")

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
<div class="footer">
    Smart Document Intelligence
    &nbsp;•&nbsp;
    AI-Powered PDF Analysis
</div>
""",
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
<div class="footer">
    Smart Document Intelligence
    &nbsp;•&nbsp;
    AI-Powered PDF Analysis
</div>
""",
        unsafe_allow_html=True
    )