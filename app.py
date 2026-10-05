import streamlit as st
from pypdf import PdfReader
from huggingface_hub import InferenceClient


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student AI Assistant",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# HUGGING FACE CONFIG
# =========================================================

client = InferenceClient(
    api_key=st.secrets["HF_TOKEN"],
    provider="auto"
)

MODEL = "openai/gpt-oss-120b"


# =========================================================
# HEADER
# =========================================================

st.title("🎓 Student AI Assistant")

st.write(
    "Your AI study assistant for summarizing lectures "
    "and answering questions from your study material."
)

st.divider()


# =========================================================
# INPUT METHOD
# =========================================================

tab_pdf, tab_text = st.tabs(
    ["📄 Upload PDF", "📝 Paste Text"]
)


pdf_text = ""
text_input = ""


# =========================================================
# PDF UPLOAD
# =========================================================

with tab_pdf:

    uploaded_file = st.file_uploader(
        "📄 Upload your lecture PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        try:

            reader = PdfReader(uploaded_file)

            pages = []

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    pages.append(page_text)

            pdf_text = "\n\n".join(pages)

            st.success(
                f"📄 {uploaded_file.name} uploaded successfully."
            )

            st.info(
                f"📄 Number of pages: {len(reader.pages)}"
            )

        except Exception as e:

            st.error(
                f"❌ Could not read the PDF: {e}"
            )


# =========================================================
# TEXT INPUT
# =========================================================

with tab_text:

    text_input = st.text_area(
        "📝 Paste your lecture text here",
        height=250,
        placeholder="Paste your study material here..."
    )


# =========================================================
# GET CONTENT
# =========================================================

content = ""

if pdf_text.strip():

    content = pdf_text

elif text_input.strip():

    content = text_input


# =========================================================
# AI TABS
# =========================================================

tab_summary, tab_question = st.tabs(
    ["✨ Generate Summary", "💬 Ask a Question"]
)


# =========================================================
# SUMMARY
# =========================================================

with tab_summary:

    st.subheader("✨ Generate a Study Summary")

    if st.button(
        "✨ Generate Summary",
        use_container_width=True
    ):

        if not content.strip():

            st.warning(
                "⚠️ Please upload a PDF or paste some text first."
            )

        else:

            with st.spinner(
                "🤖 AI is generating your summary..."
            ):

                try:

                    prompt = f"""
You are a helpful university study assistant.

Summarize the following study material clearly and accurately.

Requirements:
- Keep the important information.
- Use clear headings.
- Use bullet points when useful.
- Explain important concepts briefly.
- Do not add information that is not in the material.
- Make the summary easy for a university student to study.

Study Material:

{content}
"""

                    response = client.chat.completions.create(

                        model=MODEL,

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful university "
                                    "study assistant."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        max_tokens=2000,

                        temperature=0.3
                    )

                    summary = (
                        response.choices[0]
                        .message
                        .content
                    )

                    st.success("✅ Summary generated successfully!")

                    st.subheader("📝 AI Summary")

                    st.markdown(summary)

                except Exception as e:

                    st.error(
                        f"❌ AI Error: {e}"
                    )


# =========================================================
# QUESTION & ANSWER
# =========================================================

with tab_question:

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "What would you like to know?"
    )

    if st.button(
        "💬 Ask AI",
        use_container_width=True
    ):

        if not content.strip():

            st.warning(
                "⚠️ Please upload a PDF or paste some text first."
            )

        elif not question.strip():

            st.warning(
                "⚠️ Please enter a question."
            )

        else:

            with st.spinner(
                "🤖 AI is thinking..."
            ):

                try:

                    prompt = f"""
You are a university study assistant.

Answer the student's question using ONLY
the study material provided below.

If the answer is not found in the material,
say clearly that the information is not available
in the provided material.

Study Material:

{content}

Student Question:

{question}
"""

                    response = client.chat.completions.create(

                        model=MODEL,

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful university "
                                    "study assistant."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        max_tokens=1000,

                        temperature=0.3
                    )

                    answer = (
                        response.choices[0]
                        .message
                        .content
                    )

                    st.success("✅ Answer generated!")

                    st.subheader("🤖 AI Answer")

                    st.markdown(answer)

                except Exception as e:

                    st.error(
                        f"❌ AI Error: {e}"
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 Student AI Assistant • Powered by Hugging Face"
)