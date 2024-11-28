from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import json
image = Image.open("/home/ozdomar/Adil/doc_gpt/vision/surya_ocr/surya_data/DECISION  CONSIGNATION RADEEF_page_13.jpg")
langs = ["ar"] # Replace with your languages - optional but recommended
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
# print(type(predictions))
# print(dir(predictions[0]))
predictions[0].model_dump_json()