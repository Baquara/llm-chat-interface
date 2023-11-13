import requests

class GenerateResponseModel:
    def __init__(self, user_message, username, botname, print_timings, in_code_block):
        self.user_message = user_message
        self.username = username
        self.botname = botname
        self.print_timings = print_timings
        self.in_code_block = in_code_block

def get_response_from_server(data):
    url = "http://127.0.0.1:8000/generate_response/"  # assuming server is running locally on port 8000
    response = requests.post(url, json=data.__dict__, stream=True)

    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=512):  # get chunks of bytes
            # decode the bytes and print them immediately
            print(chunk.decode('utf-8'), end='')
    else:
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    data = GenerateResponseModel(
        user_message="How are you?",
        username="JohnDoe",
        botname="ChatGPT",
        print_timings=True,
        in_code_block=False
    )

    get_response_from_server(data)
