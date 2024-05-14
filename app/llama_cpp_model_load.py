from model_load import ModelLoad


class ModelLoaderClass:
    def __init__(self):
        self.model = ModelLoad.krypton_chat_llamacpp_model_load()

    def get_loaded_model(self):
        return self.model
