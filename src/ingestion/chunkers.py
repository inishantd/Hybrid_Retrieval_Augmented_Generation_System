import re
from typing import List

from src.config import config
from src.ingestion.schemas import Document, Chunk, ChunkMetadata


class ChunkingEngine:
    """
    A grab bag of document chunking strategies, from dumb and simple to
    structure aware. Pick whichever fits the source format.

    Strategies:
        1. Fixed-size chunking          
        2. Structure-aware markdown     
        3. Recursive chunking           
        4. Structure-aware recursive    

    For production use, `production_chunk()` 
    """

    # 1. Fixed-size chunking

    @staticmethod
    def fixed_size_chunk(document: Document, chunk_size: int = None, chunk_overlap: int = None) -> List[Chunk]:
        """
        Slide a fixed-size window over the text with some overlap between
        consecutive chunks. Doesn't care about words, sentences, or anything
        else - it just cuts at character offsets.

        e.g. chunk_size=1500, chunk_overlap=300:
            chunk 0 -> chars 0:1500
            chunk 1 -> chars 1200:2700
            chunk 2 -> chars 2400:3900
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        text = document.page_content
        if not text.strip():
            return []

        chunks = []
        chunk_index = 0
        step = chunk_size - chunk_overlap
        start = 0

        while start < len(text):
            slice_text = text[start:start + chunk_size].strip()
            if slice_text:
                chunks.append(
                    Chunk(
                        id=f"{document.id}-chunk-{chunk_index}",
                        page_content=slice_text,
                        metadata=ChunkMetadata(
                            source_path=document.metadata.source_path,
                            file_type=document.metadata.file_type,
                            chunk_index=chunk_index,
                            parent_document_id=document.id,
                        ),
                    )
                )
                chunk_index += 1
            start += step

        return chunks


    # 2. Structure-aware markdown chunking
  
    @staticmethod
    def structure_aware_markdown_chunk(document: Document) -> List[Chunk]:
        """
        Split on markdown headers (#, ##, ###, ...) and prepend the most
        recent header to each chunk so we don't lose context once the
        heading itself is gone from the chunk.
        """
        text = document.page_content
        if not text.strip():
            return []

        # Capture headers as their own elements so we can track "current section"
        sections = re.split(r"(^#+\s+.*$)", text, flags=re.MULTILINE)

        chunks = []
        chunk_index = 0
        current_header = "Introduction"

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if section.startswith("#"):
                current_header = section
                continue

            chunks.append(
                Chunk(
                    id=f"{document.id}-markdown-{chunk_index}",
                    page_content=f"{current_header}\n\n{section}",
                    metadata=ChunkMetadata(
                        source_path=document.metadata.source_path,
                        file_type=document.metadata.file_type,
                        chunk_index=chunk_index,
                        parent_document_id=document.id,
                        custom_attributes={"markdown_header": current_header},
                    ),
                )
            )
            chunk_index += 1

        return chunks


    # 3. Recursive chunking

    @staticmethod
    def recursive_chunk(document: Document, chunk_size: int = None, chunk_overlap: int = None) -> List[Chunk]:
        """
        Split on the most "natural" boundary that keeps pieces under
        chunk_size, falling back progressively:

            paragraph -> line -> sentence -> word -> raw character

        Much less likely to slice a sentence in half than fixed_size_chunk.
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        text = document.page_content.strip()
        if not text:
            return []

        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
        pieces = ChunkingEngine._recursive_split(text, chunk_size, separators)

        return ChunkingEngine._create_chunks_from_pieces(
            document=document,
            pieces=pieces,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="recursive",
        )

    # 4. Structure-aware recursive chunking (recommended for production)

    @staticmethod
    def structure_aware_recursive_chunk(
        document: Document, chunk_size: int = None, chunk_overlap: int = None
    ) -> List[Chunk]:
        """
        The best-of-both approach: find the document's markdown sections
        first, then only fall back to recursive splitting for sections that
        are too big to fit in one chunk. Small sections stay intact.
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        text = document.page_content.strip()
        if not text:
            return []

        all_chunks = []
        chunk_index = 0

        for header, content in ChunkingEngine._extract_sections(text):
            if not content.strip():
                continue

            contextualized = f"{header}\n\n{content.strip()}" if header else content.strip()

            # Section fits in one chunk as-is
            if len(contextualized) <= chunk_size:
                all_chunks.append(
                    Chunk(
                        id=f"{document.id}-struct-rec-{chunk_index}",
                        page_content=contextualized,
                        metadata=ChunkMetadata(
                            source_path=document.metadata.source_path,
                            file_type=document.metadata.file_type,
                            chunk_index=chunk_index,
                            parent_document_id=document.id,
                            custom_attributes={
                                "section_title": header,
                                "chunking_strategy": "structure_recursive",
                            },
                        ),
                    )
                )
                chunk_index += 1
                continue

            # Section is too big - fall back to recursive splitting
            pieces = ChunkingEngine._recursive_split(
                contextualized,
                chunk_size,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
            )
            recursive_chunks = ChunkingEngine._create_chunks_from_pieces(
                document=document,
                pieces=pieces,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy="structure_recursive",
                start_index=chunk_index,
                section_title=header,
            )
            all_chunks.extend(recursive_chunks)
            chunk_index += len(recursive_chunks)

        return all_chunks


    # Helpers

    @staticmethod
    def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    
        text = text.strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        if not separators:
            return [text[i:i + chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i:i + chunk_size].strip()]

        separator = separators[0]
        parts = text.split(separator)

        # Separator wasn't found at all - try the next one down
        if len(parts) == 1:
            return ChunkingEngine._recursive_split(text, chunk_size, separators[1:])

        chunks = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            candidate = f"{current}{separator}{part}" if current else part

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                # Flush what we've got, then try to fit `part` on its own
                if current:
                    chunks.extend(ChunkingEngine._recursive_split(current, chunk_size, separators[1:]))
                current = part

        if current:
            chunks.extend(ChunkingEngine._recursive_split(current, chunk_size, separators[1:]))

        return chunks

    @staticmethod
    def _extract_sections(text: str) -> List[tuple[str, str]]:
        
        header_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$", flags=re.MULTILINE)
        matches = list(header_pattern.finditer(text))

        if not matches:
            return [("", text)]

        sections = []

        # Anything before the first header still counts as content
        first_start = matches[0].start()
        if first_start > 0:
            intro = text[:first_start].strip()
            if intro:
                sections.append(("Introduction", intro))

        for i, match in enumerate(matches):
            header = match.group(0).strip()
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()
            if content:
                sections.append((header, content))

        return sections

    @staticmethod
    def _create_chunks_from_pieces(
        document: Document,
        pieces: list[str],
        chunk_size: int,
        chunk_overlap: int,
        strategy: str,
        start_index: int = 0,
        section_title: str = None,
    ) -> list[Chunk]:
    

        if not pieces:
            return []

        chunks = []

        for local_index, piece in enumerate(pieces):

            piece = piece.strip()

            if not piece:
                continue

           
            # Validate individual piece
         

            if len(piece) > chunk_size:
                raise ValueError(
                    f"Recursive splitter produced a piece larger "
                    f"than chunk_size: {len(piece)} > {chunk_size}"
                )

            
            # Add overlap from previous piece
          

            if local_index > 0 and chunk_overlap > 0:

                previous_piece = pieces[local_index - 1].strip()

                # Maximum overlap that can fit while respecting
                # chunk_size.
                available_space = (
                    chunk_size
                    - len(piece)
                    - 1  # newline
                )

                actual_overlap = min(
                    chunk_overlap,
                    max(0, available_space),
                    len(previous_piece),
                )

                if actual_overlap > 0:

                    overlap_text = previous_piece[
                        -actual_overlap:
                    ].strip()

                    candidate = (
                        overlap_text
                        + "\n"
                        + piece
                    )

                    # Final safety check
                    if len(candidate) <= chunk_size:
                        piece = candidate

           
            # Metadata


            actual_index = start_index + len(chunks)

            custom_attributes = {
                "chunking_strategy": strategy,
            }

            if section_title:
                custom_attributes[
                    "section_title"
                ] = section_title

            metadata = ChunkMetadata(
                source_path=document.metadata.source_path,
                file_type=document.metadata.file_type,
                chunk_index=actual_index,
                parent_document_id=document.id,
                custom_attributes=custom_attributes,
            )

         
            # Chunk ID
         

            chunk_id = (
                f"{document.id}-"
                f"{strategy}-"
                f"{actual_index}"
            )

            chunks.append(
                Chunk(
                    id=chunk_id,
                    page_content=piece,
                    metadata=metadata,
                )
            )

        return chunks

    # Production entry point

    @staticmethod
    def production_chunk(document: Document, chunk_size: int = None, chunk_overlap: int = None) -> List[Chunk]:
        """
        The strategy you actually want to call in production: structure-aware
        recursive chunking for markdown/HTML, plain recursive chunking for
        everything else (PDF, TXT, etc).
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

        file_type = document.metadata.file_type.value.lower()

        if "markdown" in file_type or file_type in {"md", "html"}:
            return ChunkingEngine.structure_aware_recursive_chunk(
                document=document, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        return ChunkingEngine.recursive_chunk(document=document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)