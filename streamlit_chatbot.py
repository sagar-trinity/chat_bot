import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph_email_bot import app, config  # Import from the LangGraph workflow file

st.set_page_config(page_title="LangGraph Email Chatbot", page_icon="📧")

st.markdown("<h1 style='text-align: center; font-size: 20px;'>LangGraph Email Chatbot</h1>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="you are a bot")]

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = ""
            message_placeholder = st.empty()

            for event in app.stream({"messages": st.session_state.chat_history}, config, stream_mode="values"):
                chunk = event["messages"][-1].content
                response_text += chunk
                message_placeholder.markdown(response_text)

            st.session_state.chat_history.append(AIMessage(content=response_text))

            st.markdown(response_text)

