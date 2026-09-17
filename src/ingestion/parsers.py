import re
import hashlib
import logging
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.ingestion.schemas import Document, DocumentMetadata, FileType


logger = logging.getLogger(__name__)


class DocumentParserRouter:
    """
    Routes supported files to the appropriate parser and
    returns a validated Document object.

    Supported formats:
        - PDF
        - HTML / HTM
        - Markdown
        - TXT
    """

   # SUppoerted File Type
    SUPPORTED_EXTENSIONS = {
        "pdf": FileType.PDF,
        "html": FileType.HTML,
        "htm": FileType.HTML,
        "md": FileType.MARKDOWN,
        "txt": FileType.TXT,
    }
   # Text Sentitzation

    @staticmethod
    def sanitize_unicode_string(raw_text: str) -> str:
        """
        Clean extracted text to make it safe for downstream
        Pydantic validation, embeddings, vector databases,
        and LLM processing.
        """
        if not raw_text:
            return "" #Prevent Unnecessary Processing
        
        # Remove Unicode surrogate characters
        
        clean_chars = [
            char
            for char in raw_text
            if not ("\ud800" <= char <= "\udfff")
        ]

        text = "".join(clean_chars)

        # Remove invalid UTF-8 characters
        
        text = (
            text
            .encode("utf-8", errors="ignore")
            .decode("utf-8", errors="ignore")
        )

        # Remove null bytes; Some databases and processing systems don't handle embedded null bytes well.
        
        text = text.replace("\x00", "")

        # Normalize PDF bullet characters
       
        text = text.replace("", "\n- ")

        # Fix common PDF word-joining problems
        
        pattern = (
            r"([a-z])"
            r"(based|by|down|under|from|to|for|rules|procedures)"
            r"\b"
        )

        for _ in range(2):
            text = re.sub(
                pattern,
                r"\1 \2",
                text
            )

        #  Normalize excessive whitespace
        
        lines = [
            line.strip()
            for line in text.splitlines()
        ]

        text = "\n".join(
            line
            for line in lines
            if line
        )

        return text.strip()
    
    # Hashing 
   
    @staticmethod #Usefull for Deduplication
    def calculate_content_hash(content: str) -> str:
        """
        Generate SHA-256 hash of document content.

        Used for document deduplication.
        """

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
 
    # PDF Parser
  
    def _parse_pdf(self, file_path: Path) -> str:
        """
        Extract text from all PDF pages.
        """

        try:
            reader = PdfReader(str(file_path))

            pages_text = []

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):
                page_text = page.extract_text()

                if page_text:
                    pages_text.append(
                        f"\n[Page {page_number}]\n"
                        f"{page_text}"
                    )

            return "\n".join(pages_text)

        except Exception as err:
            raise ValueError(
                f"Failed to extract text from PDF "
                f"'{file_path}': {err}"
            ) from err
   
    # HTML Parser
 
    def _parse_html(self, file_path: Path) -> str:
        """
        Extract visible text from HTML.
        """

        try:
            with file_path.open(
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:

                html_content = file.read()

            soup = BeautifulSoup(
                html_content,
                "html.parser"
            )

            # Remove elements that generally don't
            # contribute useful document content.
            for element in soup(
                ["script", "style", "noscript"]
            ):
                element.decompose()

            return soup.get_text(
                separator="\n"
            )

        except Exception as err:
            raise ValueError(
                f"Failed to parse HTML "
                f"'{file_path}': {err}"
            ) from err
  
    # Text ? Markdown parser

    def _parse_text(self, file_path: Path) -> str:
        """
        Read TXT or Markdown files as UTF-8 text.
        """

        try:
            return file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        except Exception as err:
            raise ValueError(
                f"Failed to read text file "
                f"'{file_path}': {err}"
            ) from err
    
    # Documnet Title

    @staticmethod
    def extract_document_title(
        file_path: Path,
        content: str
    ) -> str:
        """
        Attempt to determine a useful document title.

        For Markdown, use the first H1 heading if available.
        Otherwise, fall back to the filename.
        """

        # Markdown H1
        for line in content.splitlines():

            line = line.strip()

            if line.startswith("# "):
                return line[2:].strip()

        # Fallback: filename without extension
        return file_path.stem
        
    # Main 
    
    def process_file(
        self,
        file_path: str | Path
    ) -> Document:
        """
        Parse a supported file and return a validated
        Document object.
        """

        # 1. Convert input to Path
    
        path = Path(file_path)
    
        # 2. Validate file existence
       
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        # 3. Make sure it is actually a file
      
        if not path.is_file():
            raise ValueError(
                f"Expected a file but received: {path}"
            )
 
        # 4. Extract extension

        extension = path.suffix.lower().lstrip(".")
     
        # 5. Validate supported extension

        if extension not in self.SUPPORTED_EXTENSIONS:

            raise ValueError(
                f"Unsupported file type '.{extension}'. "
                f"Supported types: "
                f"{', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        file_type = self.SUPPORTED_EXTENSIONS[
            extension
        ]

        logger.info(
            "Parsing %s file: %s",
            file_type.value,
            path
        )
       
        # 6. Route to appropriate parser
    
        if extension == "pdf":

            raw_text = self._parse_pdf(path)

        elif extension in {"html", "htm"}:

            raw_text = self._parse_html(path)

        elif extension in {"md", "txt"}:

            raw_text = self._parse_text(path)

        else:
            # This should never happen because of the
            # supported-extension validation above.
            raise ValueError(
                f"No parser available for '.{extension}'"
            )

        # 7. Sanitize extracted text

        sanitized_content = (
            self.sanitize_unicode_string(raw_text)
        )

        # 8. Validate extracted content

        if not sanitized_content:
            raise ValueError(
                f"No usable text could be extracted "
                f"from '{path}'"
            )

        # 9. Generate content hash

        content_hash = self.calculate_content_hash(
            sanitized_content
        )

        # 10. Extract title

        document_title = self.extract_document_title(
            path,
            sanitized_content
        )

        # 11. Create metadata
 
        metadata = DocumentMetadata(
            source_path=str(path),
            file_type=file_type,
            document_title=document_title,
            content_hash=content_hash,
        )
     
        # 12. Create validated Document
     
        document = Document(
            id=content_hash[:16],
            page_content=sanitized_content,
            metadata=metadata,
        )

        logger.info(
            "Successfully parsed document: %s | "
            "characters=%d | hash=%s",
            path,
            len(sanitized_content),
            content_hash[:12],
        )

        return document