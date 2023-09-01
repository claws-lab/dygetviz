import requests

def download_file_from_google_drive(file_id, destination):
    r"""Downloads the content of a google drive URL to a specific folder.

    Args:
        file_id (str): The URL. Get this from the Google Drive share link
        destination (str): The filename for the downloaded file.
    """

    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":

    file_id = "1gPbI37ZbIRHJTN5w9MVvs4YZj5N1Ntlf"  # Get this from the Google Drive share link
    destination = "data"  # Local path where you want to save the file
    download_file_from_google_drive(file_id, destination)
