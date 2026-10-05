from pypdf import PdfReader
from huggingface_hub import InferenceClient

HF_TOKEN = input("Enter your Hugging Face token: ")

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)

MODEL = "openai/gpt-oss-120b"

pdf_path = "dml_env/programming.pdf"

reader = PdfReader(pdf_path)

pages = []

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        pages.append(page_text)

content = "\n\n".join(pages)

print("Number of pages:", len(reader.pages))
print(content[:2000])

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
            "content": "You are a helpful university study assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=2000,
    temperature=0.3
)

summary = response.choices[0].message.content

print("\nAI Summary:")
print(summary)

question = input("\nWhat would you like to know? ")

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
            "content": "You are a helpful university study assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=1000,
    temperature=0.3
)

answer = response.choices[0].message.content

print("\nAI Answer:")
print(answer)