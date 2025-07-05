import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Import the new loading and embedding functions
from utils import load_and_prepare_data, load_persisted_vector_store, get_custom_retriever_function, PERSIST_DIRECTORY

# --- Configuration ---
CSV_FILE_PATH = "insurance_policies_sample_100_final.csv"
GEMINI_MODEL_NAME = "gemini-2.5-pro"
TEMPERATURE = 0

# --- Environment Setup ---
load_dotenv()

print("--- Initializing Insurance Chatbot System Components ---")

# --- Step 1: Load data for policy_id_dict ---
all_documents, policy_id_dict = load_and_prepare_data(CSV_FILE_PATH)

# --- Step 2: Load Vector Store ---
vector_store = load_persisted_vector_store()

# --- Step 3: Configure Google Gemini LLM ---
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=TEMPERATURE)
print(f"Google Gemini LLM '{llm.model}' initialized successfully.")

# --- Step 4: Design the Prompt with Guardrails ---
SYSTEM_MESSAGE_TEMPLATE = """
You are an **Insurance Policy Query Assistant** trained to help users with details from their insurance policies. \
You must follow the strict guardrails below to ensure privacy, accuracy, and responsible AI behavior. \
Always respond in a polite, professional tone. Keep replies concise unless clarification is needed.

**Role & Scope**
- Only assist with **existing policy information** such as:
    - `coverage_amount`
    - `premium`
    - `renewal_date`
- You must **not** assist with:
    - Claims, cancellations, or new policy purchases
    - Personal or sensitive data such as names, addresses, or phone numbers

**Greetings**
- If user greets (e.g., "hi", "hello"), respond:
    > Hello! I’m your Insurance Policy Assistant. I can help you with questions about your policy’s coverage, \
premium, or renewal date. Please provide your policy ID or ask a specific question.
-  **Compliance Disclaimer:** No.

**Valid Policy Query**
- If a valid policy ID is provided (e.g., POL001) **and** the question asks about coverage, premium, or renewal:
- Respond with accurate information from `{context}`.
-  **Compliance Disclaimer:** No.

**Sensitive Data Guardrail**
- If the user asks for sensitive info (`customer_name`, `policy_type`, etc.), respond:
    > I'm sorry, I cannot share personal or sensitive information like customer names or policy types \
for privacy and security reasons.
-  **Compliance Disclaimer:** Yes.

**Out-of-Scope Queries**
- If the user asks about cancellations, claims, buying a policy, or unrelated topics:
    > I’m only able to assist with existing policy details like coverage, premium, and renewal dates. \
For anything else, please contact your insurance advisor or visit the official website.
-  **Compliance Disclaimer:** Yes.

**Policy Not Found or Ambiguous Input**
- If policy ID is invalid or not found in `{context}`:
    > I couldn’t find any policy with that ID. Please check the number and try again.
-  **Compliance Disclaimer:** Yes.

- If the query is vague (e.g., “Tell me about my insurance”):
    > Could you please provide a valid policy ID so I can help with accurate details?
-  **Compliance Disclaimer:** No.

**Fallback Response**
- If the query does not match any category, respond politely:
    > I'm not sure how to help with that. Please provide a valid insurance-related question.
- **Compliance Disclaimer:** Yes.

**Compliance Disclaimer**
- Include the disclaimer:
    > **Please consult an insurance advisor for detailed guidance.**

---

Use the following context to answer the user's query:
**{context}**
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
print("Chatbot Prompt defined with guardrails.")

# --- Step 5: Create the LangChain Chain ---

# First, create a chain that takes chat history and a question, and generates a standalone question for retrieval.
history_aware_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a concise standalone search query for the retriever, considering the chat history if necessary. Only return the search query itself, no other text."),
    ]
)

# get_custom_retriever_function is the specific tool it uses for the actual document lookup once the query is rephrased.
history_aware_retriever = create_history_aware_retriever(
    llm,
    get_custom_retriever_function(vector_store, policy_id_dict),
    history_aware_prompt
)
print("History-aware retriever created.")

# Create the chain that combines the retrieved documents with the prompt and LLM to answer the question.
document_chain = create_stuff_documents_chain(llm, prompt)

# Combine the history-aware retriever and the document chain into the final conversational retrieval chain.
conversational_retrieval_chain = create_retrieval_chain(
    history_aware_retriever,
    document_chain
)
print("Conversational retrieval chain created.")


# --- Step 6: Implement the Chatbot Loop ---
print("\n--- Insurance Policy Query Assistant Ready ---")
print("Type 'exit' or 'quit' to end the conversation.")

chat_history = []

while True:
    user_query = input("\n[You]: ")
    if user_query.lower() in ["exit", "quit"]:
        print("[Chatbot]: Goodbye!")
        break

    # Invoke the conversational retrieval chain
    response = conversational_retrieval_chain.invoke(
        {"input": user_query, "chat_history": chat_history}
    )

    # The 'answer' key contains the generated response from the LLM
    chatbot_response = response["answer"]
    print(f"[Chatbot]: {chatbot_response}")

    # Update chat history for the next turn
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=chatbot_response))