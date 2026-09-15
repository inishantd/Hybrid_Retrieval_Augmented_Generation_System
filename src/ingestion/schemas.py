import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#File Type
class FileType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "md"
    HTML = "html"
    TXT = "txt"


#Document MetaData
class DocumentMetadata(BaseModel):
    """Metadata associated with a source document."""

    model_config = ConfigDict(extra="forbid") #Don't allow fields that we haven't defined.

    source_path: str = Field(
        ...,
        min_length=1,
        description="Path or identifier of the source file."
    )

    file_type: FileType = Field(
        ...,
        description="Type of source document."
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC ingestion timestamp."
    )

    document_title: str | None = Field(
        default=None,
        description="Title of the source document."
    )

    source_url: str | None = Field(
        default=None,
        description="URL of the source document, if available."
    )

    content_hash: str | None = Field(
        default=None,
        min_length=1,
        description="Hash of document content used for deduplication."
    )

    custom_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional extensible metadata."
    )

# Document 
class Document(BaseModel):
    """Container representing a parsed source document."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        description="Unique document identifier."
    )

    page_content: str = Field(
        ...,
        min_length=1,
        description="Extracted plaintext content."
    )

    metadata: DocumentMetadata = Field(
        ...,
        description="Document metadata."
    )

#Cunk MetaData
class ChunkMetadata(DocumentMetadata):
    """Metadata used to track document chunk lineage."""

    chunk_index: int = Field(
        ...,
        ge=0,
        description="Sequential index of chunk inside parent document."
    )

    parent_document_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the parent document."
    )

    page_number: int | None = Field(
        default=None,
        ge=1,
        description="Source page number."
    )

    section_title: str | None = Field(
        default=None,
        description="Section or heading containing the chunk."
    )

    start_char: int | None = Field(
        default=None,
        ge=0,
        description="Starting character position in the source document."
    )
    end_char: int | None = Field(
        default=None,
        ge=0,
        description="Ending character position in the source document."
    )

#Cunk
class Chunk(BaseModel):
    """Text chunk used for sparse and dense indexing."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        description="Unique chunk identifier."
    )

    page_content: str = Field(
        ...,
        min_length=1,
        description="Text content of the chunk."
    )

    metadata: ChunkMetadata = Field(
        ...,
        description="Chunk lineage and source metadata."
    )

