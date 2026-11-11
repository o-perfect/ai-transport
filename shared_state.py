"""st.session_state keys shared by the map (app.py) and the chat beside it
(app_chat.py). Input is natural language only, so data flows chat -> map.

AGENT_ROUTES  GeoJSON FeatureCollection from the agent's latest compare_routes
              call (its "_geometry"), baseline first. LineString coordinates are
              [lon, lat]; feature properties: role ("baseline" | "alternative"),
              label, origin, destination, distance_nm, sea_days, total_cost_usd,
              chokepoints. Its "scenario" member lists the chokepoints the agent
              treated as blocked, war_risk and vessel_cannot_use. Removed when
              the user clears the map.
AGENT_BUSY    True while an agent turn runs. Set by the chat's submit callback,
              so it is already set when the page is drawn; buttons are disabled
              meanwhile, since any click would rerun the page and cancel the turn.
"""

AGENT_ROUTES = "agent_routes"
AGENT_BUSY = "agent_busy"
