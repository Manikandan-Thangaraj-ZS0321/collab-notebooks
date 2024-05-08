import torch
import gc

from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from paddleocr import PaddleOCR
from transformers import AutoProcessor, VisionEncoderDecoderModel
from transformers import StoppingCriteriaList
from nougat_extraction import StoppingCriteriaScores
from PIL import Image


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
        finally:
            gc.collect()
            torch.cuda.empty_cache()

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
        finally:
            gc.collect()
            torch.cuda.empty_cache()

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

    @staticmethod
    def text_extraction_krypton(image_path: str):
        try:
            processor, text_krypton_model = TextExtraction.krypton_text_model_load()
            pixel_values = processor(images=Image.open(image_path), return_tensors="pt").pixel_values
            device = "cuda" if torch.cuda.is_available() else "cpu"
            text_krypton_model.to(device)
            outputs = text_krypton_model.generate(
                pixel_values.to(device),
                min_length=1,
                max_length=3584,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
            )
            generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
            ocr_result = processor.post_process_generation(generated, fix_markdown=False)
            return ocr_result
        except Exception as ex:
            raise ex
        finally:
            gc.collect()
            torch.cuda.empty_cache()


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
