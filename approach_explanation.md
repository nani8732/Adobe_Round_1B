
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
Adobe Hackathon Round 1B – Approach Explanation
Problem Overview
The task for Round 1B is to act as an intelligent document analyst. Given a persona definition, a job-to-be-done, and a collection of documents (PDFs), the system must extract and prioritize the most relevant sections and subsections to support the persona’s objective.

High-Level Architecture
The solution is modular and performs the following key steps:

Document Parsing:

For each PDF document, we extract a structured outline (title + headings) using logic built in Round 1A.
Text blocks are segmented and associated with headings to form logically coherent sections.
Persona & Task Understanding:

The persona is parsed either from a .json or .txt file.
The job-to-be-done is read as a text input.
This information defines the user’s intent and relevance context.
Relevance Scoring:

Each section’s content is compared against the persona's intent using:
Keyword Overlap: Matching tokens from persona focus areas and task description.
Semantic Similarity (if model is available): Using Sentence-BERT (all-MiniLM-L6-v2) to compute cosine similarity between the section and the persona+job query.
A combined score ranks the importance of each section.
Sub-section Refinement:

For the top-ranked sections, we extract the first few meaningful sentences to provide a concise preview or "refined snippet".
Final Output:

A structured JSON is generated containing:
Metadata (documents, persona, task, timestamp)
Ranked section list with titles, page numbers, and ranks
Subsection analysis with brief refined content
Model Details
SentenceTransformer: We use all-MiniLM-L6-v2, a compact BERT-based model optimized for semantic search. It's ~90MB and ideal for fast CPU inference.
The model is cached locally inside /app/models/ to support offline execution and reduce runtime.
Offline & Docker Compatibility
The solution does not rely on any internet access.
All models and dependencies are either installed during Docker build or cached locally.
The container processes each collection from /app/input and saves results to /app/output.
Key Strengths
Modular and extensible for future use.
Handles multiple test cases automatically.
Falls back to keyword-only scoring if the model is unavailable (e.g., in restricted environments).
Compliant with execution time, model size (<1GB), and architecture constraints.
Limitations
Does not yet perform OCR for scanned PDFs.
Assumes text-based PDFs and extractable structure.
May miss relevance in documents with irregular formatting or unconventional headings.
Conclusion
This solution intelligently surfaces key document insights based on user intent, making large PDFs more navigable and goal-oriented. It’s fast, extensible, and aligns well with the goals of Adobe’s “Connecting the Dots” challenge.