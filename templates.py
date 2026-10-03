"""Starter templates shown in the builder's "New agent" menu."""

TEMPLATES = [
    {
        "label": "Blank",
        "config": {
            "name": "my-agent",
            "description": "",
            "system_prompt": "You are a helpful assistant.",
            "tools": [],
        },
    },
    {
        "label": "Coding assistant",
        "config": {
            "name": "coder",
            "description": "Reads, writes and runs code in its workspace.",
            "system_prompt": (
                "You are a careful software engineer working in a project workspace. "
                "Explore the files before changing them, make focused edits, and run the code "
                "or tests to verify your work. Summarise what you changed when you finish."
            ),
            "tools": ["list_files", "read_file", "write_file", "edit_file", "bash"],
        },
    },
    {
        "label": "Web researcher",
        "config": {
            "name": "researcher",
            "description": "Fetches web pages and writes a sourced report.",
            "system_prompt": (
                "You research questions by fetching relevant web pages. Cite the URL for every "
                "claim. When you have enough information, write a concise report to report.md "
                "and give the user a short summary."
            ),
            "tools": ["http_get", "write_file"],
        },
    },
    {
        "label": "Custom tool example",
        "config": {
            "name": "weather-bot",
            "description": "Shows how to write a custom Python tool.",
            "system_prompt": "You answer weather questions using the get_weather tool.",
            "tools": [],
            "custom_tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "City name"}},
                        "required": ["city"],
                    },
                    "code": (
                        "import json, urllib.parse, urllib.request\n\n"
                        "def run(args):\n"
                        "    city = urllib.parse.quote(args['city'])\n"
                        "    url = f'https://wttr.in/{city}?format=j1'\n"
                        "    with urllib.request.urlopen(url, timeout=15) as r:\n"
                        "        cur = json.load(r)['current_condition'][0]\n"
                        "    return f\"{cur['temp_C']}°C, {cur['weatherDesc'][0]['value']}\"\n"
                    ),
                }
            ],
        },
    },
]
