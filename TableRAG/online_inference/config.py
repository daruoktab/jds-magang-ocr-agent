# Konfigurasi LLM: semua komponen memakai VLM lokal qwen3.5:4b via Ollama
# (endpoint OpenAI-compatible http://localhost:11434/v1).
v3_config = {
    "url": "http://localhost:11434/v1",
    "model": "qwen3.5:4b",
    "api_key": "ollama"
}

# URL service SQL (Flask interface.py di offline_data_ingestion_and_query_interface)
sql_service_url = 'http://localhost:5000/get_tablerag_response'


config_mapping = {
    "v3": v3_config
}
