from ollama import Client

client = Client(host="http://127.0.0.1:11434")

response = client.embed(
    model="nomic-embed-text",
    input="Hello world"
)

print(response)