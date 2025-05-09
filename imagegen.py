import requests
import io
from PIL import Image


API_TOKEN=""

API_URL="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

headers={
    "Authorization":f"Bearer {API_TOKEN}"
}

while True:

    prompt=input("Enter a description for the image you'd like to generate: ")

    # making a request
    response=requests.post(API_URL,headers=headers,json={"inputs":prompt})

    if response.status_code==200:
        image=Image.open(io.BytesIO(response.content))
        fileName=input("Enter the name you want of the generated image file to save it: ")
        image.save(f"{fileName}.png")
        print(f"Image successfully generated and saved as {fileName}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    userChoice=input("Do you want to generate another image? (yes/no): ")
    if userChoice!='yes':
        print("Exiting the program.")
        break


# pip install requests
# python -m pip install requests
# pip show requests
