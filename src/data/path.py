from abc import ABC, abstractmethod


class PathProvider(ABC):

    @staticmethod
    @abstractmethod
    def board() -> str: pass

    @staticmethod
    @abstractmethod
    def models() -> str: pass

    @staticmethod
    @abstractmethod
    def data() -> str: pass

    @classmethod
    def create_model_path(cls, name: str, number: int = 0) -> str:
        return f'{cls.models()}/{name}_{str(number).zfill(6)}.pt'


class DGXPath(PathProvider):

    homa = "/home/hakatenco"
    ausland = "/raid/data"

    @staticmethod
    def board() -> str:
        return  "/home/ibespalov/pomoika/"

    @staticmethod
    def models() -> str:
        return DGXPath.ausland + "/hakatenco/saved_models"

    @staticmethod
    def data() -> str:
        return DGXPath.ausland


class Paths:

    default: PathProvider = DGXPath







