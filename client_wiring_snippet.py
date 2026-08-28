# --- Snippet: how enterprise_client.py should build StdioServerParameters ---
#
# Wherever the client currently loops over `config["servers"]` and creates a
# StdioServerParameters for each one, it needs to also build an `env` dict
# from that server's YAML entry and pass it through. StdioServerParameters
# accepts an `env` argument that gets set as the subprocess's environment.

import os
from mcp import StdioServerParameters


def build_server_params(server_cfg: dict) -> StdioServerParameters:
    """
    server_cfg is one entry from servers.yaml's `servers:` list, e.g.:
      {
        "name": "knowledge",
        "path": "knowledge_server.py",
        "enabled": true,
        "description": "...",
        "embed_model": "nomic-embed-text",
        "log_path": "logs/knowledge_server.log",
      }
    """
    env = os.environ.copy()  # subprocess still needs PATH, etc.

    if "embed_model" in server_cfg:
        env["KNOWLEDGE_EMBED_MODEL"] = server_cfg["embed_model"]

    if "log_path" in server_cfg:
        # Prefix with the server name so this generalizes cleanly if
        # hr_server.py / ticketing_server.py later read their own
        # HR_LOG_PATH / TICKETING_LOG_PATH the same way.
        env_key = f"{server_cfg['name'].upper()}_LOG_PATH"
        env[env_key] = server_cfg["log_path"]

    return StdioServerParameters(
        command="python",
        args=[server_cfg["path"]],
        env=env,
    )