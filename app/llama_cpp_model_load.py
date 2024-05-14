from model_load import ModelLoad


class ModelLoaderClass:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        # ... (Your model loading code)
        return ModelLoad.krypton_chat_llamacpp_model_load()

    def get_loaded_model(self):
        return self.model
