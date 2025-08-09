# gen_ai_projects

This repository showcases a collection of projects and experiments leveraging **Generative AI**, **Large Language Models (LLMs)**, and related technologies. It includes examples of multi-agent systems, custom model implementations, prompt engineering techniques, and data processing for AI tasks.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Project Demos](#project-demos)  
  - [Multi-Agent System](#multi-agent-system)  
  - [Fashion MNIST CNN Classifier](#fashion-mnist-cnn-classifier)  
  - [MNIST CNN Classifier](#mnist-cnn-classifier)  
  - [MNIST Dense Classifier](#mnist-dense-classifier)  
  - [Customer Churn Prediction ANN](#customer-churn-prediction-ann)  
  - [University Admission Prediction ANN](#university-admission-prediction-ann)  
  - [Movie Recommender (Vector Search)](#movie-recommender-vector-search)  
  - [FAQ Text Generation (LSTM)](#faq-text-generation-lstm)  
  - [Career Guide Prompt Template](#career-guide-prompt-template)  
  - [Few-Shot Prompting with LLMChain](#few-shot-prompting-with-llmchain)  
  - [ReAct Calculator Search Agent](#react-calculator-search-agent)  
  - [Stock Analysis LangGraph Agent](#stock-analysis-langgraph-agent)  
  - [MCQ Generator](#mcq-generator)  
  - [Simple Azure LLM Query](#simple-azure-llm-query)  
  - [Azure Embedding Vector Export](#azure-embedding-vector-export)  
  - [Document Tag Similarity](#document-tag-similarity)  
  - [FAISS Vector Search Demo](#faiss-vector-search-demo)  
  - [Parking Data Profiling Report](#parking-data-profiling-report)
- [Contributing](#contributing)

---

## Features

This repository demonstrates various applications of Generative AI and Machine Learning, including:

- **Multi-Agent Systems**: Orchestrating multiple specialized agents for complex task execution using LangGraph.
- **Deep Learning Models**: Implementing Convolutional Neural Networks (CNNs) and Artificial Neural Networks (ANNs) for image classification and prediction tasks.
- **Natural Language Processing (NLP)**: Utilizing sentence transformers for text similarity, generating text with LSTMs, and prompt engineering for specific tasks.
- **Vector Databases and Search**: Employing FAISS for efficient similarity search and building recommendation systems.
- **LLM Integration**: Interacting with LLMs (specifically Azure OpenAI) for various tasks like question answering, summarization, and tool usage.
- **Data Profiling**: Generating comprehensive reports on datasets using `ydata-profiling`.
- **Streamlit Applications**: Building interactive web applications for AI models.

---

## Project Structure
```
gen_ai_projects/
├── Multi-Agent Project/
│ ├── agents/
│ │ ├── init.py
│ │ ├── clinical_assistant.py
│ │ └── leave_scheduling.py
│ ├── core/
│ │ ├── init.py
│ │ ├── router.py
│ │ └── state.py
│ ├── utils/
│ │ └── db_setup.py
│ ├── main.py
│ ├── .env (example)
│ └── README.md
├── fashion_mnist_cnn_classifier.ipynb
├── mnist_cnn_classifier.ipynb
├── mnist_dense_classifier.ipynb
├── customer_churn_prediction_ann.ipynb
├── university_admission_prediction_ann.ipynb
├── movie_recommender_vector_search.py
├── faq_text_generation_lstm.ipynb
├── programming_language_expert_prompt.py
├── rag_chatbot_faiss_streamlit.py
├── react_calculator_search_agent.py
├── stock_analysis_langgraph_agent.py
├── mcq_generator.py
├── simple_azure_llm_query.py
├── azure_embedding_vector_export.py
├── document_tag_similarity.py
├── faiss_vector_search_demo.py
├── parking_data_profiling_report.ipynb
├── placement_perceptron_classifier.ipynb
├── Multi-Agent System.py
├── requirements.txt
└── README.md
```


---

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd gen_ai_projects
    ```

2. **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` includes all necessary libraries like `langchain`, `langchain-openai`, `langchain-community`, `langgraph`, `python-dotenv`, `requests`, `sqlite3`, `tensorflow`, `streamlit`, `pandas`, `numpy`, `scikit-learn`, `mlxtend`, `sentence-transformers`, `PyPDF2`, `fpdf`, `ydata-profiling`, `seaborn`, `matplotlib`, `ipython`)*

4. **Set up Azure OpenAI Credentials**:  
    Create a `.env` file in the root directory and add your credentials:
    ```dotenv
    AZURE_OPENAI_API_BASE="YOUR_AZURE_OPENAI_ENDPOINT"
    AZURE_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_API_VERSION="2024-12-01-preview"
    AZURE_OPENAI_API_NAME="YOUR_DEPLOYMENT_NAME"
    TEXT_EMBEDDING_API_BASE="YOUR_TEXT_EMBEDDING_ENDPOINT"
    TEXT_EMBEDDING_API_NAME="YOUR_TEXT_EMBEDDING_DEPLOYMENT_NAME"
    ```

5. **Initialize Database (for Multi-Agent System)**:
    ```bash
    cd Multi-Agent\ Project
    python utils/db_setup.py
    cd ..
    ```

---

## Project Demos

### Multi-Agent System
- **Description**: A system where agents specialize in tasks like clinical assistance and leave scheduling, coordinated by a router.
- **Location**: `Multi-Agent Project/`
- **Run**:
    ```bash
    python "Multi-Agent Project/main.py"
    ```
- **Key Files**: `main.py`, `agents/clinical_assistant.py`, `agents/leave_scheduling.py`, `core/router.py`, `core/state.py`, `utils/db_setup.py`

### Fashion MNIST CNN Classifier
- **Description**: CNN for classifying Fashion MNIST dataset images.
- **Location**: `fashion_mnist_cnn_classifier.ipynb`

### MNIST CNN Classifier
- **Description**: CNN for handwritten digit classification.
- **Location**: `mnist_cnn_classifier.ipynb`

### MNIST Dense Classifier
- **Description**: Fully connected neural network for MNIST dataset.
- **Location**: `mnist_dense_classifier.ipynb`

### Customer Churn Prediction ANN
- **Description**: ANN for predicting customer churn.
- **Location**: `customer_churn_prediction_ann.ipynb`

### University Admission Prediction ANN
- **Description**: ANN for predicting admission chances.
- **Location**: `university_admission_prediction_ann.ipynb`

### Movie Recommender (Vector Search)
- **Description**: Movie recommendation using sentence embeddings & cosine similarity.
- **Location**: `movie_recommender_vector_search.py`

### FAQ Text Generation (LSTM)
- **Description**: LSTM-based text generation.
- **Location**: `faq_text_generation_lstm.ipynb`

### Career Guide Prompt Template
- **Description**: Prompt template for a career guide assistant.
- **Location**: `career_guide_prompt_template.py`

### Few-Shot Prompting with LLMChain
- **Description**: Few-shot prompting using LangChain.
- **Location**: `few_shot_prompt_llmchain.py`

### ReAct Calculator Search Agent
- **Description**: ReAct agent for calculations & web search.
- **Location**: `react_calculator_search_agent.py`

### Stock Analysis LangGraph Agent
- **Description**: LangGraph agent for stock analysis.
- **Location**: `stock_analysis_langgraph_agent.py`

### MCQ Generator
- **Description**: Streamlit app to generate MCQs from PDFs.
- **Run**:
    ```bash
    streamlit run mcq_generator.py
    ```
- **Key File**: `mcq_generator.py`

### Simple Azure LLM Query
- **Description**: Query an Azure OpenAI LLM.
- **Location**: `simple_azure_llm_query.py`

### Azure Embedding Vector Export
- **Description**: Export embeddings from Azure OpenAI.
- **Location**: `azure_embedding_vector_export.py`

### Document Tag Similarity
- **Description**: Tag-based document similarity with sentence transformers.
- **Location**: `document_tag_similarity.py`

### FAISS Vector Search Demo
- **Description**: Vector storage & search with FAISS.
- **Location**: `faiss_vector_search_demo.py`

### Parking Data Profiling Report
- **Description**: Data profiling with `ydata-profiling`.
- **Location**: `parking_data_profiling_report.ipynb`

---

## Contributing

Contributions are welcome! Please submit pull requests or open issues for improvements, bug fixes, or new project ideas.
