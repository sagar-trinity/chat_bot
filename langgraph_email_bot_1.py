import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import openai
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import constant
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.pydantic_v1 import BaseModel, Field
from langgraph.pregel.retry import RetryPolicy

# Embedding model and LLM setup
embeddings = AzureOpenAIEmbeddings(
    model=constant.AZURE_OPENAI_API_EMBEDDING_MODEL,
    azure_deployment=constant.AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME,
    api_key=constant.AZURE_OPENAI_KEY,
    openai_api_version=constant.OPENAI_API_VERSION,
    azure_endpoint=constant.AZURE_OPENAI_ENDPOINT,
    openai_api_type=constant.OPENAI_API_TYPE
)

openai.api_type = constant.OPENAI_API_TYPE
openai.api_version = constant.OPENAI_API_VERSION
openai.api_key = constant.AZURE_OPENAI_KEY
openai.api_base = constant.OPENAI_API_BASE
llm = AzureChatOpenAI(deployment_name=constant.AZURE_OPENAI_DEPLOYMENT_NAME, temperature=0, cache=False, openai_api_key=openai.api_key, openai_api_version=openai.api_version)

model = AzureChatOpenAI(model_name="gpt-4o", streaming=True)
memory = MemorySaver()

# Define tools
@tool
def mail(to: str, subject: str, body: str):
    """Send an email using Gmail SMTP."""
    msg = MIMEMultipart()
    msg['From'] = 'sg23official@gmail.com'
    msg['To'] = to
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    mailserver = smtplib.SMTP('smtp.gmail.com', 587)
    mailserver.starttls()
    mailserver.login('sg23official@gmail.com', 'xyjb cvkv wajh ywkb')
    mailserver.sendmail('sg23official@gmail.com', to, msg.as_string())
    mailserver.quit()
    return "email sent"

@tool
def distance(place1: str, place2: str) -> str:
    """Returns the distance in kilometers between two places by name."""
    geolocator = Nominatim(user_agent="distance_tool")
    location1 = geolocator.geocode(place1)
    location2 = geolocator.geocode(place2)
    if not location1 or not location2:
        return f"Could not find one or both locations: '{place1}' or '{place2}'."
    coords_1 = (location1.latitude, location1.longitude)
    coords_2 = (location2.latitude, location2.longitude)
    distance_km = geodesic(coords_1, coords_2).kilometers
    return f"The distance between {place1} and {place2} is approximately {distance_km:.2f} km."

@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a given city using OpenWeatherMap API."""
    api_key = "abbe146a0b8b5752e7bac8cb38b6d109"
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        if response.status_code != 200 or "main" not in data:
            return f"Could not fetch weather data for {city}."
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The weather in {city} is {desc} with a temperature of {temp}°C."
    except Exception as e:
        return f"Error while fetching weather: {str(e)}"

# Tool metadata for vector store
tool_metadata = [
    {"name": "mail", "description": "Send an email using Gmail SMTP.", "tool": mail},
    {"name": "distance", "description": "Calculate distance between two cities.", "tool": distance},
    {"name": "get_weather", "description": "Get current weather using OpenWeather API.", "tool": get_weather}
]

# Vector store setup
documents = [
    Document(page_content=tool["description"], metadata={"tool_name": tool["name"]})
    for tool in tool_metadata
]
vector_store = FAISS.from_documents(documents, embeddings)

class QueryForTools(BaseModel):
    query: str = Field(..., description="Query for additional tools.")

def select_tools(state: MessagesState):
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        query = last_message.content
    else:
        system = SystemMessage(
            content="Given this conversation, generate a query for additional tools."
        )
        input_messages = [system] + state["messages"]
        response = llm.bind_tools([QueryForTools], tool_choice=True).invoke(input_messages)
        query = response.tool_calls[0]["args"]["query"]

    tool_documents = vector_store.similarity_search(query)
    selected_tool_names = [doc.metadata["tool_name"] for doc in tool_documents]
    selected_tools = [tool["tool"] for tool in tool_metadata if tool["name"] in selected_tool_names]
    state["selected_tools"] = selected_tools  # Save to state for use in ToolNode
    return {}

def call_model(state: MessagesState):
    bound_model = model.bind_tools(state.get("selected_tools", []))
    response = bound_model.invoke(state["messages"])
    return {"messages": response}

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "action"

class DynamicToolNode:
    def __call__(self, state: MessagesState):
        dynamic_node = ToolNode(state["selected_tools"])
        return dynamic_node.invoke(state)

workflow = StateGraph(MessagesState)
workflow.add_node("select_tools", select_tools, retry=RetryPolicy(max_attempts=2))
workflow.add_node("agent", call_model)
workflow.add_node("action", DynamicToolNode())
workflow.add_edge(START, "select_tools")
workflow.add_edge("select_tools", "agent")
workflow.add_conditional_edges("agent", should_continue, ["action", END])
workflow.add_edge("action", "agent")

app = workflow.compile(checkpointer=memory)

SYSTEM_PROMPT = "you are a helpful assistant"
chat_history = [SystemMessage(content=SYSTEM_PROMPT)]

config = {
    "configurable": {"thread_id": "user_session_1"},
    "recursion_limit": 10
}

__all__ = ["app", "config"]
