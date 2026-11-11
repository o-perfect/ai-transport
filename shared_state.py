"""st.session_state keys shared by the map page (app_planner.py) and the chat
page (app_chat.py). Both run as pages of app.py, so they see the same state.

AGENT_ROUTES  chat -> map. GeoJSON FeatureCollection of the routes from the
              agent's latest compare_routes call (its "_geometry"), baseline
              first. LineString coordinates are [lon, lat]; feature properties:
              role ("baseline" | "alternative"), label, origin, destination,
              distance_nm, chokepoints. Removed when the user clears it.
SCENARIO      map -> chat. Names of the chokepoints closed on the map, as
              PortWatch names where one matches (e.g. "Suez Canal"), else the
              map's own name. The chat adds them to the prompt as the scenario.
"""

from agent.network import PASSAGE_TO_CHOKEPOINT

AGENT_ROUTES = "agent_routes"
SCENARIO = "map_scenario"


def scenario_names(closed):
    """Map registry Chokepoints (core.chokepoints) to the names the agent uses."""
    return [PASSAGE_TO_CHOKEPOINT.get(cp.searoute_passage, cp.name) for cp in closed]
