# Insurance Policy Query Assistant

## Project Overview

This project implements an intelligent chatbot designed to assist users with inquiries regarding insurance policy details. Leveraging Google's Generative AI (Gemini 2.5 Pro) and the LangChain framework, the chatbot provides information on policy coverage, premiums, and renewal dates by querying a structured dataset of sample insurance policies.

A key feature of this chatbot is its robust set of guardrails and a custom retrieval mechanism. It prioritizes direct policy ID lookups for precise answers and strictly adheres to privacy rules, avoiding sensitive data disclosure and redirecting out-of-scope queries.

## Features

* **Policy Detail Retrieval:** Provides specific details like `coverage_amount`, `premium`, and `renewal_date` for individual policies.
* **Direct Policy ID Lookup:** Efficiently retrieves policy information based on exact policy IDs (e.g., `POL001`) using a custom retriever.
* **Conversational Memory:** Utilizes chat history to understand contextual follow-up questions.
* **Data Protection Guardrails:** Strictly prevents the disclosure of sensitive information such as `customer_name` or `policy_type`.
* **Out-of-Scope Redirection:** Politely redirects users for queries related to cancellations, claims, purchasing new policies, or legal advice.
* **Policy ID Requirement:** Enforces the provision of a valid policy ID for detailed information.
* **Vector Database Persistence:** Stores and loads policy embeddings using ChromaDB for fast retrieval without re-indexing on every run.
* **Clear Disclaimers:** Includes compliance disclaimers where appropriate.

## Technologies Used

* **Python 3.9+**
* **LangChain:** Framework for building LLM applications.
* **Google Generative AI (Gemini 2.5 Pro):** For natural language understanding and generation.
* **Google Generative AI Embeddings:** For converting text into numerical vectors.
* **ChromaDB:** Lightweight, in-memory vector database for similarity search and persistence.
* **python-dotenv:** For managing API keys and environment variables.
* **pandas:** (Implicitly used by `CSVLoader` if the CSV processing relies on it, though not directly imported in the provided snippets).

## Setup and Installation

Follow these steps to set up and run the chatbot locally:

### 1. Clone the Repository

```bash
git clone https://github.com/Nishanth456/insurance-policy-query-assistant.git
cd insurance-policy-query-assistant
```
### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
**Note**: If you don't have a requirements.txt file, create one in your project root with the following content:

### 4. Set Up Google API Key
Obtain your GOOGLE_API_KEY from the Google AI Studio or Google Cloud Console.

Create a file named .env in the root directory of your project and add your API key:

```bash
GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

### 5. Prepare Data and Create Vector Database
You need a CSV file containing your insurance policy data. Ensure it's named insurance_policies_sample_100_final.csv and placed in the project root.

Run the utils.py script once to load the data, create embeddings, and persist the vector database. This will create a chroma_db directory in your project.

```bash
python utils.py
```

**Important**: This step must be completed successfully before running the chatbot. If you update your insurance_policies_sample_100_final.csv file, you'll need to run this command again to re-index the data.

### 6. Run the Chatbot
After the vector database has been created (step 5), you can start the chatbot application:

```Bash
python chatbot.py
```

**Usage**
Once the chatbot is running, you can interact with it via your terminal.

## Project Structure
```bash
.
├── chatbot.py                 # Main chatbot application logic
├── dataset_generator.py       # Code to generate the sample dataset
├── utils.py                   # Utility functions for data loading, embeddings, and vector store management
├── insurance_policies_sample_100_final.csv # Sample insurance policy data
├── .env                       # Environment variables (e.g., GOOGLE_API_KEY)
└── chroma_db/                 # Directory where the ChromaDB vector store is persisted
├── output/                    # Contains output screenshots
└── README.md                  # This file
└── requirements.txt           # Project dependencies
```

## Guardrails and Limitations

The chatbot operates under strict guidelines to ensure privacy and focus:

It will never disclose sensitive data like customer names or policy types.

It requires a valid policy ID for specific policy details.

It strictly avoids semantic search if a policy ID is not detected in the query.

It redirects queries related to claims, cancellations, new purchases, or legal advice.

It is designed to answer questions ONLY based on the provided insurance_policies_sample_100_final.csv data.

## License

This project is licensed under the MIT License


