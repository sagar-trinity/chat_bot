import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import constant
import openai
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from geopy.geocoders import Nominatim
from geopy.distance import geodesic	
import requests

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

@tool
def mail(to: str, subject: str, body: str):
    """Send an email using Gmail SMTP."""
    msg = MIMEMultipart()
    msg['From'] = 'sg23official@gmail.com'
    msg['To'] = to
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    mailserver = smtplib.SMTP('smtp.gmail.com', 587)
    mailserver.ehlo()
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

tools = [mail, distance, get_weather]
tool_node = ToolNode(tools)
model = AzureChatOpenAI(model_name="gpt-4o", streaming=True)
bound_model = model.bind_tools(tools)
memory = MemorySaver()

def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "action"

def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    return {"messages": response}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["action", END])
workflow.add_edge("action", "agent")

app = workflow.compile(checkpointer=memory)

SYSTEM_PROMPT = "you are a bot"
chat_history = [SystemMessage(content=SYSTEM_PROMPT)]  # Initial system message

config = {
    "configurable": {"thread_id": "user_session_1"},
    "recursion_limit": 10  # Limit cycles per interaction
}

__all__ = ["app", "config"]
