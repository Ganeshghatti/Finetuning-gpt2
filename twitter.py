import requests


def get_tweet_details(tweet_url, bearer_token):
    # Extract Tweet ID from the URL
    tweet_id = tweet_url.split("/")[-1]

    # Twitter API endpoint to get tweet details
    url = f"https://api.x.com/2/tweets/1822238167216345245"

    # Headers including the Bearer Token for authentication
    headers = {"Authorization": f"Bearer {bearer_token}"}

    # Sending a GET request to the Twitter API
    response = requests.get(url, headers=headers)

    # If the request was successful, return the JSON response
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"Unable to fetch tweet details. Status code: {response.status_code}"
        }


# Replace 'YOUR_TWEET_URL' with the actual tweet URL and 'YOUR_BEARER_TOKEN' with your Bearer Token
tweet_url = "https://x.com/elonmusk/status/1822238167216345245"
bearer_token = "AAAAAAAAAAAAAAAAAAAAADEwvQEAAAAAi92GIJOBvcCi1OVEZt%2BoShOWK1Q%3DvL1VoBxA8KjPP1e6acK8CvxLFBYh8QEeXJPt0ajAxmUjYMZToP"

# Get tweet details
tweet_details = get_tweet_details(tweet_url, bearer_token)
print(tweet_details)
