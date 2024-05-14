from model_load import ModelLoad


class ModelLoader:
    def __init__(self):
        # Your model loading logic here
        self.model = self.load_model()

    def load_model(self):
        # Replace this with your actual model loading code
        print("Loading model...")
        # ... (Your model loading logic using your specific library)
        return ModelLoad.krypton_chat_llamacpp_model_load()

    def get_model(self):
        """Returns the loaded model object."""
        return self.model
