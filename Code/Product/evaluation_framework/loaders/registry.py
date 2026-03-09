import os

try:
    import fitz  
except Exception:  
    fitz = None

try:
    import docx  
except Exception:  
    docx = None


class BaseLoader:
    def load(self, file_path: str) -> str:
        raise NotImplementedError


class TxtLoader(BaseLoader):
    def load(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


class PdfLoader(BaseLoader):
    def load(self, file_path: str) -> str:
        if fitz is None:
            raise ImportError("PyMuPDF is not installed. Install with `pip install pymupdf`.")
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts)


class DocxLoader(BaseLoader):
    def load(self, file_path: str) -> str:
        if docx is None:
            raise ImportError("python-docx is not installed. Install with `pip install python-docx`.")
        document = docx.Document(file_path)
        return "\n".join(p.text for p in document.paragraphs)


class LoaderRegistry:
    def __init__(self):
        self._loaders = {}

    def register(self, extension: str, loader: BaseLoader) -> None:
        self._loaders[extension.lower()] = loader

    def load(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        loader = self._loaders.get(ext)
        if loader is None:
            raise ValueError(f"No loader registered for extension: {ext}")
        return loader.load(file_path)


def build_default_loader_registry() -> LoaderRegistry:
    registry = LoaderRegistry()
    registry.register(".txt", TxtLoader())
    registry.register(".md", TxtLoader())
    registry.register(".pdf", PdfLoader())
    registry.register(".docx", DocxLoader())
    return registry
