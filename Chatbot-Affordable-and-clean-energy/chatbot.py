#Import Statements
import streamlit as st
from langchain.chains import ConversationChain
from langchain import PromptTemplate
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

#Initialize session states
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'input' not in st.session_state:
    st.session_state['input'] = ""

if 'stored_session' not in st.session_state:
    st.session_state["stored_session"] = []

#Define function to get user input
def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """

    input_text = st.text_input("You: ", st.session_state["input"], key="input", 
                               placeholder = "Your AI assistant here! Ask me anything ...",
                               label_visibility="hidden")
    return input_text

#New Chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

st.markdown("# GreenGenie")
st.markdown("#### Inspire Change")

#API
api = st.sidebar.text_input("API-Key", type="password")

with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
    # Option to preview memory store
    if st.checkbox("Preview memory store"):
        st.write(st.session_state.entity_memory.store)
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.entity_memory.buffer)
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
    k = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)
    temperature = st.slider('Temperature', min_value = 0.0, max_value = 1.0, step = 0.1)
    template = st.selectbox("Prompt Style",('Non-environmental','Zero-Shot','Few-Shot'))

if api:

    #Create OpenAI instace

    llm = OpenAI(
            temperature = temperature,
            openai_api_key = api,
            model_name = MODEL
            )

    #Create Conv Memory
    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k = k)

    #Template
    templates = {"Zero-Shot" : """You are a environmental activist assistant to a human, powered by a large language model trained by OpenAI.

You are designed to be able to assist  related to climate change, environment and weather forecasting, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics related to climate chalenge and weather forecasting. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a range of topics of climate change and weather forecasting.

Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist. Whatever the question maybe you try your best to relate it to the topics you have and if there's no change then you can guide them in the right direction.

You must act like an environmental activist who's really passionate about the environment and life
    
Be bold to tell the answer directly also in a lovely manner
    
Give a very detailed answer with more data to them.
    
Any links must be in a clickable format like markdown maybe.
    
If not sure about the accurate answer mention that the one you're providing is the best you have but definitely give an answer for the user
    
Give sources of where they could learn more about the solution you give to as much as possible

Context:
{entities}

Current conversation:
{history}
Last line:
Human: {input}
You:""", "Non-environmental" : """You are an assistant to a human, powered by a large language model trained by OpenAI.

You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist.

Context:
{entities}

Current conversation:
{history}
Last line:
Human: {input}
You:"""}

    #Create the Conversation Chain
    Conversation = ConversationChain(
            llm = llm,
            prompt = PromptTemplate(template = templates[template], input_variables = ["entities", "history", "input"]),
            #prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory = st.session_state.entity_memory
            )

else:
    st.error("No API found")

#New Chat cont.
st.sidebar.button("New Chat", on_click = new_chat, type='primary')
#Get the user input
user_input = get_text()

#Generate the output using the ConversationChain object and the user input, add the 
if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1,-1,-1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i])
