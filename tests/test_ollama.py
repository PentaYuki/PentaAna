import ollama

response = ollama.chat(
    model="llama3:8b",
    messages=[{
        "role": "user",
        "content": "Hãy giải thích chỉ số P/E trong chứng khoán bằng tiếng Việt, ngắn gọn."
    }]
)
print(response["message"]["content"])
