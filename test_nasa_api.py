import requests
from dotenv import dotenv_values

config = dotenv_values(".env")
apikey = config["NASA_API_KEY"]
url = "https://api.nasa.gov/mars-photos/api/v1/rovers/perseverance/photos"
param = {"api_key": apikey, "sol": 100}

response = requests.get(url, params=param)
print(response.url)