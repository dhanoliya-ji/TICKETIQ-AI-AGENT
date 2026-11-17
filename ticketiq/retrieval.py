"""Knowledge-base retrieval: chunking, TF-IDF vectors in a FAISS index, top-K search."""
import faiss
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from .classifier import tokenize


# ==================================================================================================
# Splits the markdown knowledge base into retrievable chunks.
# ==================================================================================================
# We split on the ``##`` section headings rather than on a fixed number of
# characters.  The knowledge base is written as short, self-contained policy
# sections, so a section *is* the natural unit of retrieval: it never cuts a
# rule in half, and the heading itself is useful context for the language model.

# Markdown files in the knowledge base folder that are documentation *about*
# the corpus rather than part of it. Indexing these would let developer notes
# be retrieved and quoted back to a customer as if they were policy.
EXCLUDED_FILE_NAMES = {"readme.md"}


class KnowledgeChunk:
    """One retrievable section of a knowledge base document."""

    def __init__(self, chunk_id: str, source: str, document_title: str, heading: str, text: str):
        self.chunk_id = chunk_id
        # File name the chunk came from, so answers can be traced back.
        self.source = source
        # The "# " title at the top of the document.
        self.document_title = document_title
        # The "## " heading of this section.
        self.heading = heading
        # The body text under the heading.
        self.text = text

    def searchable_text(self) -> str:
        """The text used to build the vector for this chunk.

        The headings are included because they carry strong keywords such as
        "refund window" or "escalation rules".
        """
        return self.document_title + ". " + self.heading + ". " + self.text

    def as_dict(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "document_title": self.document_title,
            "heading": self.heading,
            "text": self.text,
        }


def chunk_markdown_document(path: Path) -> list[KnowledgeChunk]:
    """Split one markdown file into one chunk per ``##`` section."""
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.split("\n")

    document_title = path.stem
    current_heading = "Introduction"
    current_lines: list[str] = []
    chunks: list[KnowledgeChunk] = []

    def flush_current_section() -> None:
        """Turn whatever has been collected so far into a chunk."""
        body = "\n".join(current_lines).strip()
        if body == "":
            return
        chunk_id = path.stem + "#" + str(len(chunks) + 1)
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                source=path.name,
                document_title=document_title,
                heading=current_heading,
                text=body,
            )
        )

    for line in lines:
        if line.startswith("## "):
            # A new section starts: store the previous one first.
            flush_current_section()
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            # The document title line.
            document_title = line[2:].strip()
        else:
            current_lines.append(line)

    # The final section has no heading after it to trigger the flush.
    flush_current_section()
    return chunks


def chunk_knowledge_base(directory: Path) -> list[KnowledgeChunk]:
    """Chunk every markdown file in the knowledge base folder."""
    if not directory.exists():
        raise FileNotFoundError("Knowledge base folder not found at " + str(directory))

    all_chunks: list[KnowledgeChunk] = []
    # sorted() keeps chunk ids stable between runs.
    for path in sorted(directory.glob("*.md")):
        # The folder's own README documents the knowledge base for developers;
        # it is not policy the agent may quote to a customer, so it is skipped.
        if path.name.lower() in EXCLUDED_FILE_NAMES:
            continue
        all_chunks.extend(chunk_markdown_document(path))

    if len(all_chunks) == 0:
        raise ValueError("No markdown chunks found in " + str(directory))
    return all_chunks


# ==================================================================================================
# The vector store: a FAISS index over the knowledge base chunks.
# ==================================================================================================
# How the pieces fit
# ------------------
# 1. Each chunk is turned into a **TF-IDF** vector by scikit-learn's
#    ``TfidfVectorizer``, which the assignment names for exactly this job
#    ("scikit-learn ... for vectorization/metrics only"). It is given this
#    project's own ``tokenize`` function as its analyser, so the vocabulary is
#    the same lower-cased, stop-word-filtered one the classifier sees.
# 2. The vectors come out L2-normalised and sparse; ``toarray`` turns them into
#    the dense ``float32`` rows FAISS indexes.
# 3. The rows go into a ``faiss.IndexFlatIP`` - a flat index scored by **inner
#    product**. Because every stored vector has length 1, the inner product of a
#    query with a chunk *is* their cosine similarity, so no separate cosine step
#    is needed.
#
# On the hand-written TF-IDF next door
# ------------------------------------
# ``app/ml/tfidf.py`` implements the same weighting from its definition, and
# ``test_hand_written_tfidf_matches_sklearn`` shows the two agree to floating
# point epsilon on the real knowledge base. That is not an accident: our term
# frequency divides by the document length where scikit-learn does not, and L2
# normalisation divides that constant straight back out again. Keeping both means
# the from-scratch version is verified against a reference rather than merely
# asserted, which is the same arrangement used for the evaluation metrics.
#
# Why ``IndexFlatIP`` and not an approximate index
# ------------------------------------------------
# ``Flat`` means exhaustive: FAISS compares the query against every stored
# vector, so the results are exact rather than approximate. An approximate index
# such as ``IVFFlat`` or ``HNSW`` only starts paying for itself at hundreds of
# thousands of vectors, and it has to be *trained* on a sample first. The index
# type is the one line that would change if this knowledge base ever grew that
# far.

class SearchHit:
    """One chunk returned by a search, together with its similarity score."""

    def __init__(self, chunk: KnowledgeChunk, score: float) -> None:
        self.chunk = chunk
        self.score = score

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = dict(self.chunk.as_dict())
        result["score"] = round(self.score, 4)
        return result


def build_vectorizer() -> TfidfVectorizer:
    """The TF-IDF settings used for the knowledge base.

    ``analyzer=tokenize`` hands scikit-learn this project's tokenizer, so it
    does no preprocessing of its own and the vocabulary matches the rest of the
    system. The remaining arguments are spelled out rather than left to default
    so the weighting is readable here instead of in scikit-learn's docs.
    """
    return TfidfVectorizer(
        analyzer=tokenize,
        norm="l2",  # unit length, which is what makes inner product = cosine
        smooth_idf=True,  # idf = log((1 + n) / (1 + df)) + 1
        sublinear_tf=False,  # plain term counts, not 1 + log(count)
    )


class FaissVectorStore:
    """A FAISS flat inner-product index over TF-IDF chunk vectors."""

    def __init__(self) -> None:
        self.vectorizer = build_vectorizer()
        self.chunks: list[KnowledgeChunk] = []
        self.index: faiss.Index | None = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def build(self, chunks: list[KnowledgeChunk]) -> "FaissVectorStore":
        """Embed every chunk and load the vectors into a FAISS index."""
        if len(chunks) == 0:
            raise ValueError("cannot build a vector store from zero chunks")

        self.chunks = list(chunks)

        documents = []
        for chunk in self.chunks:
            documents.append(chunk.searchable_text())

        # The IDF values must be learned from the corpus we are going to
        # search, which is why fit happens here and not at import time.
        sparse_matrix = self.vectorizer.fit_transform(documents)

        # FAISS wants a dense, C-contiguous float32 array.
        dense_matrix = np.asarray(sparse_matrix.todense(), dtype="float32")

        index = faiss.IndexFlatIP(dense_matrix.shape[1])
        index.add(dense_matrix)
        self.index = index

        return self

    @property
    def vocabulary(self) -> list[str]:
        """The words the index has a column for, in column order."""
        if self.index is None:
            return []
        return list(self.vectorizer.get_feature_names_out())

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Return the ``top_k`` chunks most similar to the query text."""
        if self.index is None:
            raise RuntimeError("the vector store must be built before searching")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Words the knowledge base has never seen have no column, so they are
        # simply absent from the query vector.
        query_matrix = self.vectorizer.transform([query])
        query_vector = np.asarray(query_matrix.todense(), dtype="float32")

        # FAISS cannot return more neighbours than it holds.
        wanted = min(top_k, len(self.chunks))
        scores, positions = self.index.search(query_vector, wanted)

        hits: list[SearchHit] = []
        for column in range(wanted):
            position = int(positions[0][column])
            # FAISS marks "no more results" with -1.
            if position < 0:
                continue

            score = float(scores[0][column])
            # Drop chunks with no word overlap at all rather than padding the
            # answer with irrelevant policy text.
            if score <= 0.0:
                continue

            hits.append(SearchHit(self.chunks[position], score))

        # FAISS returns best-first already; sorting again makes ties stable
        # between runs by falling back to the chunk id.
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return hits

    def size(self) -> int:
        return len(self.chunks)


# ==================================================================================================
# The retrieval stage of the pipeline: ticket text in, knowledge snippets out.
# ==================================================================================================
# Plain cosine similarity over the whole knowledge base turned out not to be
# enough. A technical outage would retrieve the feature-request document, because
# words like "request", "team" and "explain" are common to both, and the agent
# would then quote *"thank the customer for the idea"* back at someone whose
# platform was down.
#
# So retrieval is category aware: each knowledge base document declares which
# ticket category it serves, and a chunk from an unrelated document has its score
# damped before the top-K is taken. The escalation rules apply to every ticket
# and are never damped. This is ordinary metadata-aware retrieval - a document
# from the wrong topic has to be a *much* better textual match to survive, rather
# than being excluded outright.

# Which ticket category each knowledge base document serves. The key is a
# prefix of the file name, so renaming a file only means editing this table.
DOCUMENT_TOPICS: dict[str, str] = {
    "01_billing_refund_policy": "billing",
    "02_account_access_troubleshooting": "account",
    "03_technical_troubleshooting": "technical",
    "04_feature_request_handling": "feature_request",
}

# Documents that apply to every ticket, whatever its category.
UNIVERSAL_DOCUMENTS = {"05_escalation_rules"}

# How much a chunk from an unrelated document is damped. At 0.45 an off-topic
# chunk needs to be more than twice as good a textual match to outrank an
# on-topic one, which is the behaviour we want: usually ignored, but still
# reachable when it is clearly the better answer.
OFF_TOPIC_MULTIPLIER = 0.45

# How many candidates to score before re-ranking. Wide enough that damped
# on-topic chunks can still climb into the final top-K.
CANDIDATE_POOL_SIZE = 12


def topic_of_document(source_file_name: str) -> str | None:
    """Return the category a document serves, or None if it serves all of them."""
    stem = source_file_name
    if stem.endswith(".md"):
        stem = stem[: -len(".md")]

    if stem in UNIVERSAL_DOCUMENTS:
        return None
    return DOCUMENT_TOPICS.get(stem)


class KnowledgeRetriever:
    """Loads the knowledge base once and answers top-K queries from memory."""

    def __init__(self, knowledge_base_dir: Path) -> None:
        self.knowledge_base_dir = knowledge_base_dir
        self.store = FaissVectorStore()
        self.is_loaded = False

    def load(self) -> "KnowledgeRetriever":
        """Chunk every markdown document and build the vector index."""
        chunks = chunk_knowledge_base(self.knowledge_base_dir)
        self.store.build(chunks)
        self.is_loaded = True
        return self

    def retrieve(self, subject: str, body: str, category: str, top_k: int) -> list[SearchHit]:
        """Find the knowledge chunks most relevant to one ticket.

        The predicted category does two jobs here: it is added to the query
        text as a keyword hint, and it decides which documents are on topic.
        """
        if not self.is_loaded:
            raise RuntimeError("the retriever must be loaded before it can retrieve")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query = category + ". " + subject + ". " + body

        # Score a wide pool first, so the re-ranking has something to work with.
        pool_size = max(CANDIDATE_POOL_SIZE, top_k)
        candidates = self.store.search(query, pool_size)

        reranked = []
        for hit in candidates:
            topic = topic_of_document(hit.chunk.source)

            if topic is None or topic == category:
                adjusted_score = hit.score
            else:
                adjusted_score = hit.score * OFF_TOPIC_MULTIPLIER

            reranked.append(SearchHit(hit.chunk, adjusted_score))

        # Best first; the chunk id breaks ties so results stay stable.
        reranked.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return reranked[:top_k]

    def chunk_count(self) -> int:
        return self.store.size()
