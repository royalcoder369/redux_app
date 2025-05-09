from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from matplotlib import pyplot as plt

image_path="image.jpg"

try:
    # load processor and model
    processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # load and process the image
    image=Image.open(image_path).convert('RGB')
    inputs=processor(image,return_tensors='pt')

    # generate caption
    output=model.generate(**inputs,max_length=30)
    caption=processor.decode(output[0],skip_special_tokens=True)

    # print caption and show image
    print(f"Image: {image_path}")
    print(f"caption: {caption}")
    
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption)
    plt.show()

except Exception as e:
    print(f"Error: {e}")


# pip install transformers torch pillow matplotlib
