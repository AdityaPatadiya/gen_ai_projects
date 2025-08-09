from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os
import uuid
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("AZURE_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_API_NAME")

llm=AzureChatOpenAI(
    openai_api_base=api_base,
    openai_api_version=api_version,
    openai_api_key=api_key,
    deployment_name=deployment_name,
    model_name="gpt-4o",
    temperature=0.9,
    top_p=0.9,
)  # type: ignore

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # History placeholder
    ("human", "{question}"),  # User input placeholder
])

memory = ConversationBufferMemory(
    memory_key = "chat_history",
    return_messages=True,
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

# Streamlit app setup
st.set_page_config(page_title="Azure OpenAI Chatbot")
st.title("Azure OpenAI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_key" not in st.session_state:
    st.session_state.input_key = str(uuid.uuid4())  # Unique key for the input field

chat_placeholder = st.container()

# Using a form to handle input and submission
with st.form("chat_form"):
    # Dynamic key for input field
    user_input = st.text_input("Ask a question:", key=st.session_state.input_key)
    submit = st.form_submit_button("Send")

if submit and user_input:
    try:
        # Process the user input
        result = chain({"question": user_input})
        bot_response = result["text"]

        # Add to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": bot_response})

        # Clear input by resetting the input key
        st.session_state.input_key = str(uuid.uuid4())

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Display entire conversation history
with chat_placeholder:
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.divider()


# """
# (venv) PS E:\coding\NBS> streamlit run code5.py

#   You can now view your Streamlit app in your browser.

#   Local URL: http://localhost:8501
#   Network URL: http://192.168.1.12:8501

# E:\coding\NBS\venv\Lib\site-packages\langchain\chat_models\__init__.py:33: LangChainDeprecationWarning: Importing chat models from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:

# `from langchain_community.chat_models import AzureChatOpenAI`.

# To install langchain-community run `pip install -U langchain-community`.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain\chat_models\__init__.py:33: LangChainDeprecationWarning: Importing chat models from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:     

# `from langchain_community.chat_models import AzureChatOpenAI`.

# To install langchain-community run `pip install -U langchain-community`.
#   warnings.warn(
# E:\coding\NBS\code5.py:17: LangChainDeprecationWarning: The class `AzureChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import AzureChatOpenAI``.
#   llm=AzureChatOpenAI(
# WARNING! top_p is not default parameter.
#                     top_p was transferred to model_kwargs.
#                     Please confirm that top_p is what you intended.
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:174: UserWarning: As of openai>=1.0.0, Azure endpoints should be specified via the `azure_endpoint` param not `openai_api_base` (or alias `base_url`). Updating `openai_api_base` from https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:181: UserWarning: As of openai>=1.0.0, if `deployment_name` (or alias `azure_deployment`) is specified then `openai_api_base` (or alias `base_url`) should not be. Instead use `deployment_name` (or alias `azure_deployment`) and `azure_endpoint`.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:189: UserWarning: As of openai>=1.0.0, if `openai_api_base` (or alias `base_url`) is specified it is expected to be of the form https://example-resource.azure.openai.com/openai/deployments/example-deployment. Updating https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\code5.py:34: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
#   memory = ConversationBufferMemory(
# E:\coding\NBS\code5.py:39: LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
#   chain = LLMChain(
# E:\coding\NBS\venv\Lib\site-packages\langchain\chat_models\__init__.py:33: LangChainDeprecationWarning: Importing chat models from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:     

# `from langchain_community.chat_models import AzureChatOpenAI`.

# To install langchain-community run `pip install -U langchain-community`.
#   warnings.warn(
# WARNING! top_p is not default parameter.
#                     top_p was transferred to model_kwargs.
#                     Please confirm that top_p is what you intended.
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:174: UserWarning: As of openai>=1.0.0, Azure endpoints should be specified via the `azure_endpoint` param not `openai_api_base` (or alias `base_url`). Updating `openai_api_base` from https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:181: UserWarning: As of openai>=1.0.0, if `deployment_name` (or alias `azure_deployment`) is specified then `openai_api_base` (or alias `base_url`) should not be. Instead use `deployment_name` (or alias `azure_deployment`) and `azure_endpoint`.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:189: UserWarning: As of openai>=1.0.0, if `openai_api_base` (or alias `base_url`) is specified it is expected to be of the form https://example-resource.azure.openai.com/openai/deployments/example-deployment. Updating https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\code5.py:67: LangChainDeprecationWarning: The method `Chain.__call__` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
#   result = chain({"question": user_input})


# > Entering new LLMChain chain...
# Prompt after formatting:
# System: You are a helpful assistant.
# Human: what is python?

# > Finished chain.
# E:\coding\NBS\venv\Lib\site-packages\langchain\chat_models\__init__.py:33: LangChainDeprecationWarning: Importing chat models from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:     

# `from langchain_community.chat_models import AzureChatOpenAI`.

# To install langchain-community run `pip install -U langchain-community`.
#   warnings.warn(
# WARNING! top_p is not default parameter.
#                     top_p was transferred to model_kwargs.
#                     Please confirm that top_p is what you intended.
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:174: UserWarning: As of openai>=1.0.0, Azure endpoints should be specified via the `azure_endpoint` param not `openai_api_base` (or alias `base_url`). Updating `openai_api_base` from https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:181: UserWarning: As of openai>=1.0.0, if `deployment_name` (or alias `azure_deployment`) is specified then `openai_api_base` (or alias `base_url`) should not be. Instead use `deployment_name` (or alias `azure_deployment`) and `azure_endpoint`.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:189: UserWarning: As of openai>=1.0.0, if `openai_api_base` (or alias `base_url`) is specified it is expected to be of the form https://example-resource.azure.openai.com/openai/deployments/example-deployment. Updating https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain\chat_models\__init__.py:33: LangChainDeprecationWarning: Importing chat models from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:     

# `from langchain_community.chat_models import AzureChatOpenAI`.

# To install langchain-community run `pip install -U langchain-community`.
#   warnings.warn(
# WARNING! top_p is not default parameter.
#                     top_p was transferred to model_kwargs.
#                     Please confirm that top_p is what you intended.
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:174: UserWarning: As of openai>=1.0.0, Azure endpoints should be specified via the `azure_endpoint` param not `openai_api_base` (or alias `base_url`). Updating `openai_api_base` from https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:181: UserWarning: As of openai>=1.0.0, if `deployment_name` (or alias `azure_deployment`) is specified then `openai_api_base` (or alias `base_url`) should not be. Instead use `deployment_name` (or alias `azure_deployment`) and `azure_endpoint`.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:189: UserWarning: As of openai>=1.0.0, if `openai_api_base` (or alias `base_url`) is specified it is expected to be of the form https://example-resource.azure.openai.com/openai/deployments/example-deployment. Updating https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(


# > Entering new LLMChain chain...
# Prompt after formatting:
# System: You are a helpful assistant.
# Human: 10 * 3 = ?

# > Finished chain.
# E:\coding\NBS\venv\Lib\site-packages\langchain\chat_models\__init__.py:33: LangChainDeprecationWarning: Importing chat models from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:     

# `from langchain_community.chat_models import AzureChatOpenAI`.

# To install langchain-community run `pip install -U langchain-community`.
#   warnings.warn(
# WARNING! top_p is not default parameter.
#                     top_p was transferred to model_kwargs.
#                     Please confirm that top_p is what you intended.
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:174: UserWarning: As of openai>=1.0.0, Azure endpoints should be specified via the `azure_endpoint` param not `openai_api_base` (or alias `base_url`). Updating `openai_api_base` from https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:181: UserWarning: As of openai>=1.0.0, if `deployment_name` (or alias `azure_deployment`) is specified then `openai_api_base` (or alias `base_url`) should not be. Instead use `deployment_name` (or alias `azure_deployment`) and `azure_endpoint`.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:189: UserWarning: As of openai>=1.0.0, if `openai_api_base` (or alias `base_url`) is specified it is expected to be of the form https://example-resource.azure.openai.com/openai/deployments/example-deployment. Updating https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain\chat_models\__init__.py:33: LangChainDeprecationWarning: Importing chat models from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:     

# `from langchain_community.chat_models import AzureChatOpenAI`.

# To install langchain-community run `pip install -U langchain-community`.
#   warnings.warn(
# WARNING! top_p is not default parameter.
#                     top_p was transferred to model_kwargs.
#                     Please confirm that top_p is what you intended.
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:174: UserWarning: As of openai>=1.0.0, Azure endpoints should be specified via the `azure_endpoint` param not `openai_api_base` (or alias `base_url`). Updating `openai_api_base` from https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:181: UserWarning: As of openai>=1.0.0, if `deployment_name` (or alias `azure_deployment`) is specified then `openai_api_base` (or alias `base_url`) should not be. Instead use `deployment_name` (or alias `azure_deployment`) and `azure_endpoint`.
#   warnings.warn(
# E:\coding\NBS\venv\Lib\site-packages\langchain_community\chat_models\azure_openai.py:189: UserWarning: As of openai>=1.0.0, if `openai_api_base` (or alias `base_url`) is specified it is expected to be of the form https://example-resource.azure.openai.com/openai/deployments/example-deployment. Updating https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/ to https://aadit-mbok6f21-eastus2.cognitiveservices.azure.com/openai.
#   warnings.warn(


# > Entering new LLMChain chain...
# Prompt after formatting:
# System: You are a helpful assistant.
# Human: can you tell me about the air plane crash in ahmedabad?

# > Finished chain.

# here i've noticed something in the terminal that when i the first program starts it will compile or starts the llm and then when i enter the question it will answer and it shows that the entering new llmchain chain... and the prompt format and at the end Finished chain shows and again i enter the question then it just compile the llm chain and not show the answer and i need to type the question again then it will answer the question and the same thing goes on
# """
