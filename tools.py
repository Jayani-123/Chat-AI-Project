import os
import json
import re
from datetime import datetime
from typing import Dict, Any, List

import requests
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import Tool

from config import config, get_config_value
from rag import rag_query

# ------------------------------------------------------------------------------------
# Base tool clients (no routing / no if-else to decide which tool to use)
# ------------------------------------------------------------------------------------
search_tool = DuckDuckGoSearchResults(output_format="list")

# OpenWeatherMap current-conditions client (API key provided via environment)
weather_api = OpenWeatherMapAPIWrapper(
    openweathermap_api_key=os.getenv("OPENWEATHERMAP_API_KEY")
)

# ------------------------------------------------------------------------------------
# DuckDuckGo – return compact results with URLs as sources
# ------------------------------------------------------------------------------------
def duckduckgo_with_sources(query: str) -> str:
    results = search_tool.run(query) or []
    max_results = get_config_value(config, "tools.duckduckgo.max_results", default=3)

    def ok(link: str) -> bool:
        if not link or not link.startswith("http"): return False
        bad = ("bing.com/aclick", "doubleclick", "/aclk?", "adservice")
        return not any(b in link for b in bad)

    lines = []
    for i, res in enumerate([r for r in results if ok(r.get("link",""))][:max_results], 1):
        title = res.get("title", "No title")
        snippet = (res.get("snippet") or "")[:200]
        url = res.get("link")
        lines.append(f"{i}. {title}\n   Snippet: {snippet}\n   Source: {url}")
    return "Search Results:\n" + ("\n\n".join(lines) if lines else "No search results found.")


# ------------------------------------------------------------------------------------
# Weather – CURRENT conditions only
# ------------------------------------------------------------------------------------
def weather_current(query: str) -> str:
    """
    Get current weather conditions for a *location string* (e.g., 'Hobart, AU', 'Launceston').
    The agent must pass the location text directly. No cleaning, no branching here.
    """
    try:
        now_local = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
        result_text = weather_api.run(query)  # raw string from the LC wrapper
        return f"Current Weather for {query} (as of {now_local}):\n{result_text}"
    except Exception as e:
        return f"Weather error for '{query}': {e}"

# ------------------------------------------------------------------------------------
# Weather – FLEXIBLE multi-day daily forecast (Open-Meteo, 1–7 days)
# ------------------------------------------------------------------------------------
def _openmeteo_forecast_days(query: str) -> str:
    try:
        q = query.strip()

        # 1) Days: accept "3 day", "3 days", "3-day", "for 3 days"
        m_days = re.search(r"(?i)\bfor\s+(\d+)\s*[- ]?\s*days?\b", q)
        if not m_days:
            m_days = re.search(r"(?i)\b(\d+)\s*[- ]?\s*days?\b", q)
        days = int(m_days.group(1)) if m_days else 3
        days = max(1, min(days, 7))

        # 2) Place:
        # Prefer the token(s) after "for" or "in"
        m_place = re.search(r"(?i)\b(?:for|in)\s+([A-Za-z][A-Za-z\s,.\-]{1,80})$", q)
        if m_place:
            location = m_place.group(1).strip()
        else:
            # Fallback: remove common words and numbers, then tidy punctuation
            tmp = re.sub(r"(?i)\b(forecast|weather|for|in|of|the|next|day|days|today|tomorrow|week)\b", " ", q)
            tmp = re.sub(r"\d+", " ", tmp)
            # remove bullets/leading punctuation and repeated spaces
            tmp = re.sub(r"^[\-\•\*\s,.;:]+", "", tmp)
            tmp = re.sub(r"\s{2,}", " ", tmp).strip()
            location = tmp or "Hobart, AU"

        # Extra hardening: strip any leading punctuation that might still linger
        location = re.sub(r"^[\-\•\*\s,.;:]+", "", location).strip(",. ")

        # --- Geocode ---
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=15,
        ).json()

        # If that failed and user didn’t include country, try appending AU
        if not geo.get("results") and not re.search(r"(?i)\bAU\b|Australia", location):
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": f"{location}, AU", "count": 1},
                timeout=15,
            ).json()

        if not geo.get("results"):
            return f"Forecast error: couldn’t geocode '{location}'. Try 'Hobart, AU'."

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        fc = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
            },
            timeout=15,
        ).json()

        daily = fc.get("daily", {})
        times = daily.get("time", [])
        tmin = daily.get("temperature_2m_min", [])
        tmax = daily.get("temperature_2m_max", [])
        rain = daily.get("precipitation_sum", [])

        if not times or len(times) < (days + 1):
            return f"Forecast error: insufficient data for {days} days."

        lines = [f"**{days}-Day Forecast for {location}**"]
        for i in range(1, days + 1):  # skip today
            d = datetime.strptime(times[i], "%Y-%m-%d").strftime("%a, %d %b")
            lines.append(f"• {d} — Low: {tmin[i]:.1f}°C | High: {tmax[i]:.1f}°C | Rain: {rain[i]:.1f} mm")
        return "\n".join(lines)

    except Exception as e:
        return f"Forecast error for '{query}': {e}"

# ------------------------------------------------------------------------------------
# Trip Budget Planner (RAG-first, no baked-in defaults)
# ------------------------------------------------------------------------------------
_PRICE_RE = re.compile(
    r"(?:\$|aud\s*)?(\d{1,4})(?:\s*(?:per\s*(?:night|day)|/day|/night))?",
    flags=re.IGNORECASE,
)

def _first_price(text: str):
    """Return the first integer price found in text, or None."""
    m = _PRICE_RE.search(text or "")
    return int(m.group(1)) if m else None

def _shorten(text: str, max_lines: int = 3, max_chars: int = 220) -> str:
    if not text:
        return "No data found."
    brief = " ".join(text.strip().split("\n")[:max_lines])
    return (brief[:max_chars] + "…") if len(brief) > max_chars else brief

def trip_budget_planner(json_input: str) -> str:
    """
    Trip budget estimator (RAG-based).

    INPUT (string): JSON object with:
      - "place": city/area name (e.g., "Hobart")
      - "days": integer number of days (e.g., 3)

    BEHAVIOR:
      - Queries the RAG index for accommodation, food, and vehicle rental in the given place.
      - Extracts only *explicit* numeric prices found in the text (AUD assumed).
      - Computes totals only for components that have explicit prices.
      - If a component has no explicit price, it is listed without a computed subtotal.
      - No hard-coded default costs. No routing logic.

    OUTPUT:
      A concise markdown summary with bullet points and a breakdown of any computable totals.
    """
    try:
        payload: Dict[str, Any] = json.loads(json_input)
        place = str(payload.get("place", "")).strip() or "Tasmania"
        days_raw = payload.get("days", 0)

        # Normalize days to int >= 0
        try:
            days = int(days_raw)
            days = days if days >= 0 else 0
        except Exception:
            days = 0

        # Query RAG (document-first)
        stay_text   = rag_query(f"List accommodation or camping options in {place} with any AUD prices.")
        food_text   = rag_query(f"List cheap food or cafes in {place} with any AUD prices.")
        rental_text = rag_query(f"List vehicle or car rental options in {place} with any AUD prices.")

        # Extract first explicit price for each component (if any)
        stay_price   = _first_price(stay_text)
        food_price   = _first_price(food_text)
        rental_price = _first_price(rental_text)

        # Prepare short lines for display
        stay_line   = _shorten(stay_text)
        food_line   = _shorten(food_text)
        rental_line = _shorten(rental_text)

        # Build the response. Totals are computed only when we have prices.
        lines: List[str] = [
            f"🗺️ **Trip Budget & Planner for {place}{f' ({days} days)' if days else ''}**",
            "",
            f"🏕️ **Accommodation / Camping:** {stay_line}",
            f"🍽️ **Food Options:** {food_line}",
            f"🚗 **Vehicle Rentals:** {rental_line}",
            "",
            "💰 **Estimated Cost Breakdown (AUD, only where explicit prices were found):**",
        ]

        components = []
        if stay_price is not None and days:
            components.append(("Accommodation", stay_price, days))
        if food_price is not None and days:
            components.append(("Food", food_price, days))
        if rental_price is not None and days:
            components.append(("Vehicle Rental", rental_price, days))

        for label, unit, n_days in components:
            subtotal = unit * n_days
            lines.append(f"• {label}: ${unit} × {n_days} = ${subtotal}")

        if components:
            total = sum(unit * n_days for _, unit, n_days in components)
            lines.append(f"----------------------------------\n**Estimated Total: ${total} AUD**")
        else:
            lines.append("• No explicit per-day prices detected; unable to compute a total.")

        return "\n".join(lines)

    except Exception as e:
        return f"TripBudgetPlanner error: {e}"

# ------------------------------------------------------------------------------------
# LangChain Tool objects (descriptions tell the LLM exactly how to call them)
# ------------------------------------------------------------------------------------
duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    description=(
        "Web search for fresh/external info. "
        "Input: a plain English query. "
        "Output: top results with short snippets and source URLs."
    ),
    func=duckduckgo_with_sources,
)

weather_tool = Tool(
    name="WeatherTool",
    description=(
        "Get CURRENT weather conditions for a location. "
        "Input: a location string like 'Hobart, AU' or 'Launceston'. "
        "Do NOT ask for forecasts here; this tool returns *current* conditions only."
    ),
    func=weather_current,
)

WeatherForecast_tool = Tool(
    name="WeatherForecast",
    description=(
        "Get a flexible multi-day DAILY forecast (min/max temperature and precipitation) "
        "for any location using Open-Meteo. Input: a natural-language query such as "
        "'forecast for Hobart', '5-day forecast in Launceston', or '7 day weather Hobart'. "
        "The tool automatically detects the number of days (1–7) and the location."
    ),
    func=_openmeteo_forecast_days,
)

trip_budget_tool = Tool(
    name="TripBudgetPlanner",
    description=(
        "Estimate a backpacker trip budget using ONLY facts from the RAG documents. "
        "Input: a JSON string with keys, for example: "
        "{\"place\": \"Hobart\", \"days\": 3}. "
        "It will list accommodation/food/rental items found in the docs and compute totals "
        "ONLY when explicit per-day AUD prices are present. No defaults."
    ),
    func=trip_budget_planner,
)