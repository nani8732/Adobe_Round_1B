
Files
..
Drop files to upload them to session storage.
Disk
69.51 GB available
Welcome to Colab!
Access Popular LLMs via Google-Colab-AI Without an API Key
Users with Colab's paid plans have free access to most popular LLMs via google-colab-ai Python library. For more details, refer to the getting started with google colab ai.

from google.colab import ai
response = ai.generate_text("What is the capital of France?")
print(response)
Explore the Gemini API
The Gemini API gives you access to Gemini models created by Google DeepMind. Gemini models are built from the ground up to be multimodal, so you can reason seamlessly across text, images, code, and audio.

How to get started?

Go to Google AI Studio and log in with your Google account.
Create an API key.
Use a quickstart for Python, or call the REST API using curl.
Discover Gemini's advanced capabilities

Play with Gemini multimodal outputs, mixing text and images in an iterative way.
Discover the multimodal Live API (demo here).
Learn how to analyze images and detect items in your pictures using Gemini (bonus, there's a 3D version as well!).
Unlock the power of Gemini thinking model, capable of solving complex task with its inner thoughts.
Explore complex use cases

Use Gemini grounding capabilities to create a report on a company based on what the model can find on internet.
Extract invoices and form data from PDF in a structured way.
Create illustrations based on a whole book using Gemini large context window and Imagen.
To learn more, check out the Gemini cookbook or visit the Gemini API documentation.

Colab now has AI features powered by Gemini. The video below provides information on how to use these features, whether you're new to Python, or a seasoned veteran.

Thumbnail for a video showing 3 AI-powered Google Colab features
What is Colab?
Colab, or "Colaboratory", allows you to write and execute Python in your browser, with

Zero configuration required
Access to GPUs free of charge
Easy sharing
Whether you're a student, a data scientist or an AI researcher, Colab can make your work easier. Watch Introduction to Colab or Colab Features You May Have Missed to learn more, or just get started below!

Getting started
The document you are reading is not a static web page, but an interactive environment called a Colab notebook that lets you write and execute code.

For example, here is a code cell with a short Python script that computes a value, stores it in a variable, and prints the result:


[ ]
seconds_in_a_day = 24 * 60 * 60
seconds_in_a_day
86400
To execute the code in the above cell, select it with a click and then either press the play button to the left of the code, or use the keyboard shortcut "Command/Ctrl+Enter". To edit the code, just click the cell and start editing.

Variables that you define in one cell can later be used in other cells:


[ ]
seconds_in_a_week = 7 * seconds_in_a_day
seconds_in_a_week
604800
Colab notebooks allow you to combine executable code and rich text in a single document, along with images, HTML, LaTeX and more. When you create your own Colab notebooks, they are stored in your Google Drive account. You can easily share your Colab notebooks with co-workers or friends, allowing them to comment on your notebooks or even edit them. To learn more, see Overview of Colab. To create a new Colab notebook you can use the File menu above, or use the following link: create a new Colab notebook.

Colab notebooks are Jupyter notebooks that are hosted by Colab. To learn more about the Jupyter project, see jupyter.org.

Data science
With Colab you can harness the full power of popular Python libraries to analyze and visualize data. The code cell below uses numpy to generate some random data, and uses matplotlib to visualize it. To edit the code, just click the cell and start editing.

You can import your own data into Colab notebooks from your Google Drive account, including from spreadsheets, as well as from Github and many other sources. To learn more about importing data, and how Colab can be used for data science, see the links below under Working with Data.


[ ]
import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)

Colab notebooks execute code on Google's cloud servers, meaning you can leverage the power of Google hardware, including GPUs and TPUs, regardless of the power of your machine. All you need is a browser.

For example, if you find yourself waiting for pandas code to finish running and want to go faster, you can switch to a GPU Runtime and use libraries like RAPIDS cuDF that provide zero-code-change acceleration.

To learn more about accelerating pandas on Colab, see the 10 minute guide or US stock market data analysis demo.

Machine learning
With Colab you can import an image dataset, train an image classifier on it, and evaluate the model, all in just a few lines of code.

Colab is used extensively in the machine learning community with applications including:

Getting started with TensorFlow
Developing and training neural networks
Experimenting with TPUs
Disseminating AI research
Creating tutorials
To see sample Colab notebooks that demonstrate machine learning applications, see the machine learning examples below.

More Resources
Working with Notebooks in Colab
Overview of Colab
Guide to Markdown
Importing libraries and installing dependencies
Saving and loading notebooks in GitHub
Interactive forms
Interactive widgets

Working with Data
Loading data: Drive, Sheets, and Google Cloud Storage
Charts: visualizing data
Getting started with BigQuery
Machine Learning
These are a few of the notebooks related to Machine Learning, including Google's online Machine Learning course. See the full course website for more.

Intro to Pandas DataFrame
Intro to RAPIDS cuDF to accelerate pandas
Getting Started with cuML's accelerator mode
Linear regression with tf.keras using synthetic data

Using Accelerated Hardware
TensorFlow with GPUs
TPUs in Colab

Featured examples
Retraining an Image Classifier: Build a Keras model on top of a pre-trained image classifier to distinguish flowers.
Text Classification: Classify IMDB movie reviews as either positive or negative.
Style Transfer: Use deep learning to transfer style between images.
Multilingual Universal Sentence Encoder Q&A: Use a machine learning model to answer questions from the SQuAD dataset.
Video Interpolation: Predict what happened in a video between the first and the last frame.

[3]
0s
!git clone https://github.com/KrishnaCodeCrafter/Adobe_Round1B.git
Cloning into 'Adobe_Round1B'...
remote: Enumerating objects: 74, done.
remote: Counting objects: 100% (74/74), done.
remote: Compressing objects: 100% (63/63), done.
remote: Total 74 (delta 6), reused 70 (delta 5), pack-reused 0 (from 0)
Receiving objects: 100% (74/74), 17.26 MiB | 36.82 MiB/s, done.
Resolving deltas: 100% (6/6), done.
Colab paid products - Cancel contracts here
Adobe India Hackathon - Round 1B: Persona-Driven Document Intelligence
Challenge Overview
This project provides a solution for Round 1B of the Adobe India Hackathon, "Connecting the Dots." The core challenge is to build an intelligent document analysis system capable of extracting and prioritizing the most relevant sections from a collection of PDF documents. This analysis is driven by a specified user persona and their particular job-to-be-done.

This solution leverages the document outlining capabilities developed in Round 1A and extends them with advanced text analysis and semantic understanding to deliver contextually relevant insights.

Features
Multi-Document Processing: Handles a collection of 3-10 related PDF documents.

Persona-Driven Analysis: Integrates a persona definition (role, expertise, focus areas) and a specific job description to guide information extraction.

Intelligent Section Extraction: Utilizes hierarchical PDF outline (from Round 1A) to accurately segment documents into logical sections.

Relevance Scoring & Ranking: Scores document sections based on their semantic and keyword-based relevance to the provided persona and job-to-be-done. Sections are then ranked by importance.

Sub-Section Analysis: Provides refined, concise text snippets from the most relevant sections, offering granular insights.

JSON Output: Generates structured JSON output conforming to the challenge's specified format (challenge1b_output.json).

Dockerized Solution: Packaged in a Docker container for consistent, isolated, and offline execution, meeting all hackathon environment requirements.

🚀 Getting Started
These instructions will get a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
Docker Desktop installed and running on your Windows, macOS, or Linux machine.

Git (optional, but recommended for cloning this repository).

Installation and Execution
Clone the repository (or set up the directory manually):

git clone [https://github.com/KrishnaCodeCrafter/Adobe_Round1B.git](https://github.com/KrishnaCodeCrafter/Adobe_Round1B.git)
cd Adobe_Round1B
(If you're not using Git, manually create the directory structure and place the files.)

Place Sample Input Files: Create subdirectories within the input/ folder for your test cases (e.g., input/test_case_1/). Inside each test case folder, create a documents/ folder and place your PDF files there. Also, create persona_definition.json (or .txt) and job_to_be_done.txt for each test case as per the structure above.

Build the Docker Image: This step will download necessary dependencies and the pre-trained NLP model (all-MiniLM-L6-v2) and bundle everything into a Docker image. This process requires an active internet connection.

docker build --platform linux/amd64 -t myr1bsolution:latest .
(This command will take some time, especially on the first run, as it downloads the Python base image and the NLP model.)

Run the Docker Container: Once the image is built, execute the following command. This will run your solution, process all test cases found in the input/ directory, and save the results to the output/ directory. The container will run in an isolated environment without internet access.

For PowerShell:

  docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none myr1bsolution:latest
For Command Prompt (CMD):

  docker run --rm -v "%cd%\input:/app/input" -v "%cd%\output:/app/output" --network none myr1bsolution:latest
Check Results: After the command finishes, your processed JSON output files will be available in the output/ directory of your local Adobe_Round1B folder. Each test case will have its own JSON file (e.g., test_case_1_academic_research_output.json).

Methodology & Approach
This solution for Round 1B builds modularly on the document parsing foundation established in Round 1A.

Document Sectioning (Leveraging Round 1A):

The pdf_parser.py module, containing the core logic from Round 1A, is used to extract the hierarchical outline (Title, H1, H2, H3) from each PDF.

The extract_document_sections function in main_round1b.py then takes this outline and performs a detailed block-level text extraction using PyMuPDF (fitz). It intelligently segments the document's full text content, associating each paragraph or block of text with its most relevant preceding heading based on their vertical positions. This ensures coherent sections for analysis.

Input Parsing:

The load_persona_and_job function dynamically reads the persona_definition.json (or .txt) and job_to_be_done.txt files for each individual test case, making the solution flexible for diverse inputs.
Relevance Scoring:

This is the core "intelligence" of Round 1B. For each extracted document section, a relevance score is calculated based on two primary mechanisms:

Enhanced Keyword Matching: Keywords derived from the job_to_be_done and the persona's focus_areas are extracted. Sections containing these keywords receive a base score.

Semantic Similarity (NLP-powered): A pre-trained SentenceTransformer model, all-MiniLM-L6-v2, is utilized. A combined query embedding is generated from the persona's description, role, and the job-to-be-done. This query is then compared (using cosine similarity) against the embedding of each document section's text. This provides a more nuanced understanding of conceptual relevance beyond exact keyword matches. The semantic similarity score is heavily weighted.

Section Ranking:

All processed sections, each with its calculated relevance score, are sorted in descending order. A final 1-based importance_rank is then assigned based on this sorted order, indicating the most pertinent sections.
Sub-Section Analysis:

For the top N (currently 10) most relevant sections, a granular refined_text is extracted. For simplicity and to adhere to constraints, this currently involves extracting the first few sentences of the relevant section. In a more advanced scenario, this could involve extractive summarization or more targeted information extraction.
Constraints Adherence
This solution is designed to strictly adhere to the challenge constraints:

CPU Only: All processing is done on the CPU; no GPU dependencies are used.

Model Size ($\\le 1$$\\le 1$GB): The all-MiniLM-L6-v2 Sentence Transformer model is approximately 90MB, well within the 1GB limit.

Processing Time ($\\le 60$$\\le 60$ seconds for 3-5 documents): The use of efficient PDF parsing libraries (PyMuPDF) and a relatively small NLP model helps keep processing times optimized.

Offline Execution: The SentenceTransformer model is pre-downloaded and saved into the Docker image during the build process. The docker run command uses --network none, ensuring no external network calls are made during execution.

Potential Future Improvements
More Advanced Sub-Section Analysis: Implement extractive summarization techniques (e.g., TextRank, or fine-tuned smaller models) or targeted information extraction based on question-answering for refined_text.

Persona Nuance: Develop more sophisticated ways to parse and utilize persona details, such as dynamically weighting different relevance factors based on the persona's role (e.g., "Investment Analyst" might prioritize numerical data).

Hierarchical Ranking Refinement: Further refine the ranking by considering the hierarchical position of sections (e.g., H1 sections might have a base importance boost).

Error Handling and Robustness: Enhance parsing for highly complex or scanned PDFs, potentially integrating light-weight OCR if within limits.