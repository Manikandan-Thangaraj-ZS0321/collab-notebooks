from model_load import ModelLoad


def load_model():
    return ModelLoad.krypton_chat_llamacpp_model_load()  # Return the loaded model


class ModelLoaderClass:
    def __init__(self):
        self.model = load_model()  # Your model loading logic

    def get_loaded_model(self):
        return self.model
