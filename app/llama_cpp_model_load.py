from model_load import ModelLoad


class ModelLoader:
    """Singleton class to load and manage the model."""

    __instance = None  # Private attribute to store the single instance

    def __new__(cls):
        """Creates a new instance or returns the existing one."""
        if not ModelLoader.__instance:
            ModelLoader.__instance = object.__new__(cls)
        return ModelLoader.__instance

    def __init__(self):
        """Private constructor to prevent direct object creation."""
        if self.__instance is not None:
            raise Exception("Singleton class can't be instantiated directly.")
        # Your model loading logic here
        self.model = self.load_model()

    def load_model(self):
        # ... (Your model loading code)
        return ModelLoad.krypton_chat_llamacpp_model_load()

    def get_model(self):
        """Returns the loaded model object."""
        return self.model
