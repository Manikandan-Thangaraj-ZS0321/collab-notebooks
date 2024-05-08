import torch

from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from paddleocr import PaddleOCR
from transformers import AutoProcessor, VisionEncoderDecoderModel


class TextExtraction:
    def __init__(self):
        pass

    @staticmethod
    def text_extraction_xenon(file_path, model):
        try:
            doc = DocumentFile.from_images(file_path)
            output = model(doc)
            json_output = output.export()
            words = get_words(json_output)
            paragraph = ' '.join(words)
            return paragraph
        except Exception as ex:
            raise ex

    @staticmethod
    def text_extraction_argon(image_path: str, model):
        try:
            result_paddle = model.ocr(image_path, cls=True)
            extracted_text = ""
            for result in result_paddle:
                for record in result:
                    txt = record[1][0]
                    extracted_text += txt + "\n"
            return extracted_text
        except Exception as e:
            raise e

    @staticmethod
    def argon_text_model_load():
        if torch.cuda.is_available():
            return PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True)
        else:
            return PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)

    @staticmethod
    def xenon_text_model_load():
        if torch.cuda.is_available():
            return ocr_predictor(pretrained=True).cuda()
        else:
            return ocr_predictor(pretrained=True).cpu()

    @staticmethod
    def krypton_text_model_load():
        processor = AutoProcessor.from_pretrained("facebook/nougat-small")
        model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")
        return processor, model


def get_words(output):
    try:
        # page_dim = output['pages'][0]["dimensions"]
        text_coordinates = []
        for obj1 in output['pages'][0]["blocks"]:
            for obj2 in obj1["lines"]:
                for obj3 in obj2["words"]:
                    text_coordinates.append(obj3["value"])
        return text_coordinates
    except Exception as ex:
        raise ex

