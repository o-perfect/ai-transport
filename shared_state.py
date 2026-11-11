"""st.session_state keys shared by the map (app.py) and the chat beside it
(app_chat.py).

AGENT_ROUTES  chat -> map. GeoJSON FeatureCollection of the routes from the
              agent's latest compare_routes call (its "_geometry"), baseline
              first. LineString coordinates are [lon, lat]; feature properties:
              role ("baseline" | "alternative"), label, origin, destination,
              distance_nm, chokepoints. Removed when the user clears it.
SCENARIO      map -> chat. Names of the chokepoints closed on the map, as
              PortWatch names where one matches (e.g. "Suez Canal"), else the
              map's own name. The chat adds them to the prompt as the scenario.
AGENT_BUSY    chat -> map. True while an agent turn runs. Set by the chat's
              submit callback, so it is already set when the sidebar is drawn;
              the map disables its controls, since any click would rerun the
              page and cancel the turn.
"""

from agent.network import PASSAGE_TO_CHOKEPOINT

AGENT_ROUTES = "agent_routes"
SCENARIO = "map_scenario"
AGENT_BUSY = "agent_busy"


def scenario_names(closed):
    """Map registry Chokepoints (core.chokepoints) to the names the agent uses."""
    return [PASSAGE_TO_CHOKEPOINT.get(cp.searoute_passage, cp.name) for cp in closed]
