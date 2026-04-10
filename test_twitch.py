import requests

url = "https://gql.twitch.tv/gql"
headers = {
    "Client-ID": "kimne78kx3ncx6brgo4mv6wki5h1ko",
}

query = """
query {
  searchCategories(query: "Elden Ring", first: 1) {
    edges {
      node {
        name
        viewersCount
      }
    }
  }
}
"""
try:
    response = requests.post(url, headers=headers, json={"query": query})
    print(response.status_code)
    print(response.text)
except Exception as e:
    print(e)
