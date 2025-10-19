# tools.py
import os
import re
from rag import rag_query
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import Tool
from config import config, get_config_value  # Now imports work

# Initialize base tools (API key already set in config.py)
search_tool = DuckDuckGoSearchResults(output_format="list")
#weather_api = OpenWeatherMapAPIWrapper()
# Initialize weather API with key from .env
weather_api = OpenWeatherMapAPIWrapper(
    openweathermap_api_key=os.getenv("OPENWEATHERMAP_API_KEY")
    
)


def duckduckgo_with_sources(query: str) -> str:
    """Search DuckDuckGo and format results with URLs as sources."""
    try:
        results = search_tool.run(query)
        if not results:
            return "No search results found."
        max_results = get_config_value(config, "tools.duckduckgo.max_results", default=3)
        formatted_results = []
        for i, res in enumerate(results[:max_results], 1):
            title = res.get('title', 'No title')
            snippet = res.get('snippet', 'No snippet')[:200] + '...' if len(res.get('snippet', '')) > 200 else res.get('snippet', '')
            url = res.get('link', 'No URL')  # Changed from 'url'
            formatted_results.append(f"{i}. {title}\n   Snippet: {snippet}\n   Source: {url}")
        return "Search Results:\n" + "\n\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {e}"

from openmeteo_py import OWmanager, Options
from datetime import datetime, timedelta
import requests
from datetime import datetime, timedelta

def get_forecast_openmeteo(city_name: str) -> str:
    """Fetch a 3-day forecast (tomorrow + 2 more days) using Open-Meteo REST API."""
    try:
        # Convert city name → latitude/longitude
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
        geo_resp = requests.get(geo_url).json()
        if "results" not in geo_resp or len(geo_resp["results"]) == 0:
            return f" Could not locate city '{city_name}'. Try 'Hobart' or 'Launceston'."

        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]

        #  Request 7-day forecast
        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&timezone=auto"
        )
        data = requests.get(forecast_url).json()

        # 3️⃣ Prepare readable 3-day summary (days 1–3)
        daily = data["daily"]
        output_lines = [f" **3-Day Forecast for {city_name.title()}**"]

        for i in range(1, 4):
            date = daily["time"][i]
            tmin = daily["temperature_2m_min"][i]
            tmax = daily["temperature_2m_max"][i]
            rain = daily["precipitation_sum"][i]
            date_fmt = datetime.strptime(date, "%Y-%m-%d").strftime("%a, %d %b")
            output_lines.append(
                f"• **{date_fmt}** — Low: {tmin:.1f}°C | High: {tmax:.1f}°C | Rain: {rain:.1f} mm"
            )

        return "\n".join(output_lines)

    except Exception as e:
        return f" Forecast unavailable: {e}"





def weather_with_source(query: str) -> str:
    """
    Unified weather function supporting both OpenWeather (current)
    and Open-Meteo (forecast) APIs.
    """
    try:
        q_lower = query.lower()
        cleaned_query = re.sub(
            r"\b(what|is|the|current|today|tomorrow|weather|in|at|for|forecast|next|week|rain|will|it|be|day|after|like|tell|me|about)\b",
            "",
            q_lower
        )
        cleaned_query = re.sub(r"[^\w\s]", "", cleaned_query).strip().title()
        if not cleaned_query:
            cleaned_query = "Hobart, AU"

        forecast_keywords = ["tomorrow", "forecast", "next", "week", "day after"]
        is_forecast = any(word in q_lower for word in forecast_keywords)

        if is_forecast:
            print(" Using Open-Meteo for forecast:", cleaned_query)
            result = get_forecast_openmeteo(cleaned_query)
            label = "Forecast"
        else:
            print(" Using OpenWeather (LangChain) for current weather:", cleaned_query)
            result = weather_api.run(cleaned_query)
            label = "Current Weather"

        now_local = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
        return f"{label} for {cleaned_query} (as of {now_local}):\n{result}"

    except Exception as e:
        return f" Unable to fetch weather for '{query}': {e}. Try 'Hobart, AU'."



def trip_budget_planner(query: str) -> str:
    """
    Fetch recommended accommodation, food, and rentals from RAG,
    extract real prices when available, and estimate a minimum travel budget.
    """
    try:
        # --- 1️⃣ Extract place and duration ---
        place_match = re.search(r"in ([A-Za-z\s]+)", query)
        days_match = re.search(r"(\d+)\s*(day|days)", query)
        place = place_match.group(1).strip().title() if place_match else "Hobart"
        days = int(days_match.group(1)) if days_match else 3

        # --- 2️⃣ Query RAG for relevant info ---
        stay_text = rag_query(f"accommodation or camping options in {place}")
        food_text = rag_query(f"cheap food or cafes in {place}")
        rental_text = rag_query(f"vehicle or car rental in {place}")

        # --- 3️⃣ Helper: extract numeric cost from text ---
        def extract_cost(text: str, default: int) -> int:
            if not text:
                return default
            # Look for things like "$70", "AUD 50", "40 per night", "from 60"
            match = re.search(r"(?:\$|aud\s*)?(\d{2,3})", text.lower())
            return int(match.group(1)) if match else default

        # --- 4️⃣ Extract prices dynamically ---
        food_cost = extract_cost(food_text, 40)
        stay_cost = extract_cost(stay_text, 70)
        rental_cost = extract_cost(rental_text, 60)

        # --- 5️⃣ Extract short descriptive lines ---
        def summarize(text):
            if not text:
                return "No data found."
            return " ".join(text.strip().split("\n")[:2])[:180]

        stay_line = summarize(stay_text)
        food_line = summarize(food_text)
        rental_line = summarize(rental_text)

        # --- 6️⃣ Calculate totals ---
        total_food = food_cost * days
        total_stay = stay_cost * days
        total_rental = rental_cost * days
        total = total_food + total_stay + total_rental

        # --- 7️⃣ Build final formatted summary ---
        return (
            f"🗺️ **Trip Budget & Planner for {place} ({days} days)**\n\n"
            f"🏕️ **Accommodation / Camping:** {stay_line}\n"
            f"🍽️ **Food Options:** {food_line}\n"
            f"🚗 **Vehicle Rentals:** {rental_line}\n\n"
            f"💰 **Estimated Cost Breakdown:**\n"
            f"• Food: ${food_cost} × {days} = ${total_food}\n"
            f"• Accommodation: ${stay_cost} × {days} = ${total_stay}\n"
            f"• Vehicle Rental: ${rental_cost} × {days} = ${total_rental}\n"
            f"----------------------------------\n"
            f"**Estimated Total: ${total} AUD**"
        )

    except Exception as e:
        return f" Could not generate trip budget: {e}"


# Register as LangChain tool
trip_budget_tool = Tool(
    name="TripBudgetPlanner",
    description="Plan a trip by listing accommodation, food, and rental options from RAG and calculating a minimum budget.",
    func=trip_budget_planner,
)


# Create Tool instances
duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    description="Use for web facts that are not in the PDF or for fresh/live info. Return snippets+URLs. Combine this with other tools when the question has multiple parts.",
    func=duckduckgo_with_sources,
)

weather_tool = Tool(
    name="WeatherTool",
    description="Use for current weather or forecast for a location. Combine with RAGTool for broader travel answers.",
    func=weather_with_source,
)
trip_budget_tool = Tool(
    name="TripBudgetPlanner",
    description=(
        "Plans a trip by retrieving accommodation, food, and rental data "
        "from the RAG documents, extracting real prices, and estimating a total budget."
    ),
    func=trip_budget_planner,
)