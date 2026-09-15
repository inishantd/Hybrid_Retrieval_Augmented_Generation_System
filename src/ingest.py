import logging
from pathlib import Path

from src.config import config
from src.ingestion.parsers import DocumentParserRouter
from src.ingestion.chunkers import ChunkingEngine
from src.ingestion.deduplicator import ChunkDeduplicator

from src.retrieval.sparse import SparseBM25Index
from src.retrieval.dense import DenseVectorIndex


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    PDF/TXT/HTML/MD
          ↓
    Parser
          ↓
    Document
          ↓
    Chunking
          ↓
    Deduplication
          ↓
    Sparse BM25 Index
    Dense ChromaDB Index
    """

    def __init__(self):
        self.parser = DocumentParserRouter()
        self.chunker = ChunkingEngine()
        self.deduplicator = ChunkDeduplicator()

        self.sparse_index = SparseBM25Index(
            storage_path=str(config.SPARSE_INDEX_PATH)
        )
        # Load any existing sparse index so re-ingesting 
        self.sparse_index.load_index()

        self.dense_index = DenseVectorIndex()

    def ingest_file(self, file_path: str | Path) -> None:
        file_path = Path(file_path)

        logger.info("=" * 70)
        logger.info("STARTING DOCUMENT INGESTION")
        logger.info("=" * 70)


        # Parse document


        logger.info("Step 1/5: Parsing document...")

        document = self.parser.process_file(file_path)

        logger.info(
            "Document parsed successfully | title='%s' | characters=%d",
            document.metadata.document_title,
            len(document.page_content),
        )

 
        # Chunk document


        logger.info("Step 2/5: Creating chunks...")

        chunks = self.chunker.production_chunk(
            document=document,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        if not chunks:
            raise ValueError("Chunking produced zero chunks.")

        logger.info(
            "Chunking complete | chunks=%d | chunk_size=%d | overlap=%d",
            len(chunks),
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP,
        )

        #  Deduplicate chunks


        logger.info("Step 3/5: Deduplicating chunks...")

        unique_chunks = self.deduplicator.deduplicate(
            chunks=chunks,
            embedding_fn=self.dense_index.embedding_fn,
            threshold=0.95,
        )

        logger.info(
            "Deduplication complete | before=%d | after=%d | removed=%d",
            len(chunks),
            len(unique_chunks),
            len(chunks) - len(unique_chunks),
        )

        if not unique_chunks:
            raise ValueError(
                "All chunks were removed during deduplication."
            )


        # Build sparse BM25 index


        logger.info("Step 4/5: Updating sparse BM25 index...")

        self.sparse_index.index_chunks(unique_chunks)

        logger.info(
            "Sparse index ready | total_chunks=%d",
            len(self.sparse_index.indexed_chunks),
        )


        #  Build dense ChromaDB index


        logger.info("Step 5/5: Updating dense ChromaDB index...")

        self.dense_index.index_chunks(unique_chunks)

        logger.info(
            "Dense index ready | indexed_chunks=%d",
            len(unique_chunks),
        )

        logger.info("=" * 70)
        logger.info("DOCUMENT INGESTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)


if __name__ == "__main__":

    # The original PDF used to generate the golden dataset is kept locally and is
    # not committed to GitHub. To run ingestion, place the PDF in sample_data/
    # and update the filename below if using a different document.
    document_path = (
        config.BASE_DIR
        / "sample_data"
        / "Code_of_Conduct_and_Expectations_from_Employees.pdf" 
    )

    if not document_path.exists():
        raise FileNotFoundError(
            f"Document not found:\n{document_path}\n\n"
            "Place your PDF inside the sample_data directory."
        )

    pipeline = DocumentIngestionPipeline()

    pipeline.ingest_file(document_path)