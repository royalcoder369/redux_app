import requests, random, numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
coco = COCO('captions_val2017.json')

def evaluate_caption(num_images=3):
    results = [] 
    for img_id in random.sample(coco.getImgIds(), num_images):
        img_info = coco.loadImgs(img_id)[0]
        img_url = f"http://images.cocodataset.org/val2017/{img_info['file_name']}"
        gt_caption = coco.loadAnns(coco.getAnnIds(imgIds=img_id))[0]['caption']
        
        try:
            image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
            out = model.generate(**processor(image, return_tensors="pt"), max_length=20)
            gen_caption = processor.decode(out[0], skip_special_tokens=True)

            gen_words = set(gen_caption.lower().split())
            gt_words = set(gt_caption.lower().split())
            word_overlap = len(gen_words.intersection(gt_words)) / max(len(gen_words), 1)

            embs = sentence_model.encode([gen_caption, gt_caption])
            sim = np.dot(embs[0], embs[1])/(np.linalg.norm(embs[0])*np.linalg.norm(embs[1]))
            
            results.append({
                'image_url': img_url,'generated': gen_caption, 'ground_truth': gt_caption,'relevance': word_overlap, 'coherence': sim          
            })
        except Exception as e:
            print(f"Error with image {img_id}: {e}")
    return results

if __name__ == "__main__":
    for i, r in enumerate (evaluate_caption(3)):
        print(f"\nImage {i+1}: \nURL: {r['image_url']}")
        print(f"Generated: {r['generated']}")
        print(f"Ground Truth: {r['ground_truth']}")
        print(f"Relevance (word overlap): {r['relevance']:.4f}")
        print(f"Coherence: {r['coherence']:.4f}")