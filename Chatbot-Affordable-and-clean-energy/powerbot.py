import streamlit as st
from streamlit_chat import message
import os
import openai

st.set_page_config(
    page_title="PowerBot",
    page_icon=":⚡:",
)


def get_initial_message():
    prompt = """
    As an energy conservation expert, provide information on the benefits and importance of conserving energy, including the impact on the environment, society, and the economy. 
    Introduce yourself as the PowerBot, an AI chatbot which can answer any queries about energy conservation and related topics.
    Related topics about energy conservation can also include Renewable Energy Sources, Energy Efficiency, Sustainable Transportation, Sustainable Agriculture, Green Buildings, Energy Conservation Education, Energy Policies and Regulations, Energy Audits. 
    But you are not limited to these topics. You can speak about any topics which is even remotely related to energy conservation.  
    Discuss energy conservation practices that can be applied in residential, commercial, and industrial settings, such as reducing energy consumption, using renewable energy sources, and improving energy efficiency. 
    Explain the role of technology in energy conservation, including innovative solutions that are currently being developed. 
    Encourage the audience to take action to conserve energy in their daily lives and discuss the potential impact of these actions on the future of energy conservation.
    Provide accurate statistics about current affairs relating to energy conservation and if the user wants it, give step by step instructions on how to switch to renewable resources to conserve energy.
    Basically you are an energy conservation assistant that has in-depth knowledge about energy conservation and other topics related to it, but know only basic information on all other things.
    Answer the questions asked by the user even if they are not about energy conservation or related topics but remind the user that you only know basic information about this and would like to concentrate on energy conservation and related topics.
    If the topic asked by the user is remotely related to energy conservation or related topics, then elaborate on it but if not give basic information and reiterate that you are a chatbot optimized for energy conservation or related topics. 
    """
    messages=[
            {"role": "system", "content": prompt},
        ]
    return messages


def get_chatgpt_response(messages, model="gpt-3.5-turbo"):
    print("model: ", model)
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages
    )
    return  response['choices'][0]['message']['content']

def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages


openai.api_key = "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"
st.title("PowerBot")
st.markdown("#### Your friendly power guide!!")

model = "gpt-3.5-turbo"
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    

placeholder = st.empty()
query = st.text_input(" ", key = "input")

if 'messages' not in st.session_state:
    st.session_state['messages'] = get_initial_message()
if query:
    with st.spinner("Your Query Is Being Processed..."):
        messages = st.session_state['messages']
        messages = update_chat(messages, "user", query)
        response = get_chatgpt_response(messages, model)
        messages = update_chat(messages, "assistant", response)
        st.session_state.past.append(query)
        st.session_state.generated.append(response)


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))


    with st.expander("Show Message History:"):
        st.write(messages)
