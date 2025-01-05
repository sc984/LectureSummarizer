import os
import random
import re
import pdfplumber
import spacy
from flask import Flask, request, render_template_string
import nltk

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

app = Flask(__name__)

nltk.download('punkt_tab')

nlp = spacy.load("en_core_web_sm")

LANGUAGE = "english"

# HELPER FUNCTIONS

def clean_extracted_text(text):
    """
    Remove repeated characters, sequences of punctuation, and extra whitespace.
    Adjust the regexes if you have other recurring patterns to clean.
    """
    # Remove repeated single-letter patterns, like "j j j"
    text = re.sub(r"(\b\w\b(\s+)){3,}", " ", text)

    # Remove sequences of repeated punctuation or random symbols
    text = re.sub(r"([@#$%^&*()~])\1+", r"\1", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def extract_text_from_pdf(pdf_path):
    """
    Extract text from the PDF and clean it.
    """
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            page_text = clean_extracted_text(page_text)
            if page_text:
                full_text += page_text + "\n"
    return full_text.strip()

def extractive_summary(text, sentence_count=5):
    """
    Use Sumy (LSA) to create an extractive summary of `sentence_count` sentences.
    """
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    summarizer = LsaSummarizer(Stemmer(LANGUAGE))
    summarizer.stop_words = get_stop_words(LANGUAGE)

    summary_sentences = summarizer(parser.document, sentence_count)
    # Join them into one text block
    return " ".join(str(s) for s in summary_sentences)

def summarize_text(text, num_sentences=7):
    """
    Extractive summarization entry point.
    Adjust `num_sentences` to capture more or fewer details.
    """
    # Remove non-ASCII to avoid any tokenizer issues
    text = text.encode("ascii", "ignore").decode("ascii")

    # Perform extractive summary
    summary = extractive_summary(text, sentence_count=num_sentences)
    return summary

def make_bullet_points(summary_text):
    """
    Convert the summary into bullet points by splitting on periods.
    """
    # Split on '.' and filter out empty strings
    sentences = [s.strip() for s in summary_text.split('.') if s.strip()]
    # Build an HTML bullet list
    bullet_summary = "".join([f"<li>{sentence}</li>" for sentence in sentences])
    return bullet_summary

def extract_keywords(summary, top_n=10):
    """
    Extract distinct noun chunks (longer than 1 word) using spaCy, and
    return the top N by frequency. Substring filtering is done to reduce duplicates.
    """
    # Remove HTML list tags
    raw_text = summary.replace('<li>', '').replace('</li>', '')
    doc = nlp(raw_text)

    # Get noun chunks longer than 1 word
    noun_chunks = [chunk.text.strip().lower()
                   for chunk in doc.noun_chunks
                   if len(chunk.text.strip().split()) > 1]

    freq = {}
    for chunk in noun_chunks:
        freq[chunk] = freq.get(chunk, 0) + 1

    # Sort by frequency
    sorted_chunks = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # Remove near-duplicates or substrings
    unique_chunks = []
    for item, _count in sorted_chunks:
        if not any(item in uc for uc in unique_chunks):
            unique_chunks.append(item)

    return unique_chunks[:top_n]

def shorten_text(text, max_length=120):
    """
    Shorten the sentence to avoid overly long MCQ options or blanks.
    """
    text = text.strip()
    if len(text) > max_length:
        return text[:max_length].rstrip() + "..."
    return text

# Global set of used sentences
used_sentences = set()

def generate_mcq(keyword, raw_summary):
    sentences = raw_summary.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    # Filter out used sentences
    available_sentences = [s for s in sentences if s not in used_sentences]

    # Find a sentence containing the keyword
    keyword_sentences = [s for s in available_sentences if keyword.lower() in s.lower()]
    if keyword_sentences:
        chosen_sentence = random.choice(keyword_sentences)
    else:
        # If none found, fallback to any available sentence
        if available_sentences:
            chosen_sentence = random.choice(available_sentences)
        else:
            # If everything is used, fallback to all
            chosen_sentence = random.choice(sentences)

    used_sentences.add(chosen_sentence) 
    chosen_sentence = shorten_text(chosen_sentence, max_length=120)

    correct_answer = f"It relates to: '{chosen_sentence}'"
    distractors = [
        "It is unrelated to the main topic.",
        "It contradicts the central idea.",
        "It is mentioned only in passing, not as a key concept."
    ]
    random.shuffle(distractors)
    options = [correct_answer] + distractors
    random.shuffle(options)

    question_text = (
        f"<p><strong>MCQ:</strong><br>"
        f"Which of the following statements best describes <b>'{keyword}'</b>?</p>"
    )
    formatted_options = "<ol type='A'>" + "".join([f"<li>{opt}</li>" for opt in options]) + "</ol>"
    return f"<div class='question'>{question_text}{formatted_options}</div>"

def generate_fill_in_blank(keyword, raw_summary):
    sentences = raw_summary.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    # Filter out any sentences already used
    available_sentences = [s for s in sentences if s not in used_sentences]

    keyword_sentences = [s for s in available_sentences if keyword.lower() in s.lower()]
    if keyword_sentences:
        chosen_sentence = random.choice(keyword_sentences)
    else:
        if available_sentences:
            chosen_sentence = random.choice(available_sentences)
        else:
            chosen_sentence = ""

    used_sentences.add(chosen_sentence)
    chosen_sentence = shorten_text(chosen_sentence, max_length=120)

    if keyword.lower() in chosen_sentence.lower():
        # Replace only the first occurrence (case-insensitive)
        blank_sentence = re.sub(keyword, "______", chosen_sentence, count=1, flags=re.IGNORECASE)
    else:
        blank_sentence = f"The concept of ______ is crucial in this text."

    return f"<div class='question'><strong>Fill in the Blank:</strong><br><p>{blank_sentence}</p></div>"

def generate_true_false(keyword, raw_summary):
    """
    Keep T/F short and generic to avoid too much duplication.
    """
    is_true = random.choice([True, False])
    if is_true:
        statement = f"'{keyword}' is discussed as a significant concept."
    else:
        statement = f"The text states that '{keyword}' has no relevance to the topic."
    return f"<div class='question'><strong>True or False:</strong><br>{statement}</div>"

def generate_short_answer(keyword, raw_summary):
    """
    Ask a generic short-answer question about the keyword.
    """
    return (
        f"<div class='question'><strong>Short Answer:</strong><br>"
        f"Explain the role or importance of '{keyword}' based on the text.</div>"
    )

def generate_questions(keywords, bullet_summary, num_questions_per_keyword=2):
    """
    Build different question types for each keyword.
    We do a limited number of question types to avoid repetition.
    """
    question_types = ["MCQ", "FillInBlank", "TrueFalse", "ShortAnswer"]
    questions = []

    # Remove HTML tags from bullet_summary
    raw_summary = bullet_summary.replace('<li>', '').replace('</li>', '')

    for kw in keywords:
        # Randomly pick  question types 
        chosen_types = random.sample(question_types, min(num_questions_per_keyword, len(question_types)))
        for qt in chosen_types:
            if qt == "MCQ":
                q = generate_mcq(kw, raw_summary)
            elif qt == "FillInBlank":
                q = generate_fill_in_blank(kw, raw_summary)
            elif qt == "TrueFalse":
                q = generate_true_false(kw, raw_summary)
            else:  # ShortAnswer
                q = generate_short_answer(kw, raw_summary)
            questions.append(q)

    return questions

# ROUTES

@app.route('/', methods=['GET'])
def index():
    return '''
    <html>
      <head>
        <title>PDF Summarizer</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f5f5f5;
          }
          h1, h2 {
            text-align: center;
          }
          .container {
            max-width: 800px;
            margin: auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
          }
          ul {
            list-style-type: disc;
            margin-left: 20px;
          }
          .question {
            background: #fafafa;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 5px;
          }
          .question strong {
            display: block;
            margin-bottom: 5px;
          }
          a {
            color: #007BFF;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Upload a PDF</h1>
          <form action="/process_pdf" method="post" enctype="multipart/form-data">
            <input type="file" name="pdf_file" accept="application/pdf" required>
            <br><br>
            <input type="submit" value="Summarize and Generate Questions">
          </form>
        </div>
      </body>
    </html>
    '''

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    global used_sentences
    used_sentences.clear()  # Clear set

    if 'pdf_file' not in request.files:
        return "No PDF file uploaded."

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return "No selected file."

    # Save the uploaded PDF
    pdf_path = os.path.join("uploads", pdf_file.filename)
    os.makedirs("uploads", exist_ok=True)
    pdf_file.save(pdf_path)

    # 1) Extract text
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text.strip():
        return "No text found in the PDF."

    # 2) Summarize text (extractive approach)
    # Adjust the number of sentences if you want more or fewer details
    summarized_text = summarize_text(full_text, num_sentences=7)

    # 3) Convert summarized text to bullet points
    bullet_summary = make_bullet_points(summarized_text)

    # 4) Extract keywords from the bullet summary
    keywords = extract_keywords(bullet_summary)

    # 5) Generate questions based on these keywords
    questions = generate_questions(keywords, bullet_summary)

    # Create HTML response
    html_content = f"""
    <html>
      <head>
        <title>Summary and Questions</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f5f5f5;
          }}
          h1, h2 {{
            text-align: center;
          }}
          .container {{
            max-width: 800px;
            margin: auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
          }}
          ul {{
            list-style-type: disc;
            margin-left: 20px;
          }}
          .question {{
            background: #fafafa;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 5px;
          }}
          .question strong {{
            display: block;
            margin-bottom: 5px;
          }}
          p a {{
            color: #007BFF;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Summary</h1>
          <ul>
            {bullet_summary}
          </ul>
          <h2>Quiz Questions</h2>
          {''.join(questions)}
          <p style="text-align:center;"><a href="/">Upload another PDF</a></p>
        </div>
      </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == "__main__":
    app.run(debug=True)
