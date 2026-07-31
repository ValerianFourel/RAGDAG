"""Module 1 - the retrieval pipeline written as an explicit structural causal model.

The SCM under test is::

    Q ──► M1 = BM25(Q)        ┐
      └─► M2 = Dense(Q)       ├─► C = TopK(M1) ∪ TopK(M2) ──► R = CE(Q, C) ──► Y = rank(d)
                              ┘

Every arrow is a separately callable function, and :meth:`RetrievalPipeline.run`
accepts a *different* query for each stage. That is what makes the mediation
analysis in :mod:`mediation` an exact computation rather than an estimate: the
pipeline is deterministic, so freezing a stage at its baseline input and
re-running yields the path-specific effect directly, with no identification
assumptions beyond the SCM itself.

Run standalone for the baseline sanity check::

    python -m pipeline
"""

from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

import config

FirstStage = Literal["union", "bm25", "dense"]

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# sklearn's list is imported lazily to keep ``import pipeline`` cheap.
_STOPWORDS: frozenset[str] | None = None


def stopwords() -> frozenset[str]:
    """English stopword set used for tokenisation, overlap and term sampling."""
    global _STOPWORDS
    if _STOPWORDS is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        _STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
    return _STOPWORDS


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens (>=2 chars). No stemming - used for covariates."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed. Used for Jaccard overlap and injection."""
    sw = stopwords()
    return [t for t in tokenize(text) if t not in sw]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard overlap between two stopword-filtered token sets.

    Causal role: a pre-treatment covariate X. It proxies lexical affinity
    between query and document, which confounds "document contains concept c"
    with "document scores highly" in :mod:`dml_analysis`.
    """
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------- #
# Cross-encoder backend
# --------------------------------------------------------------------------- #
class CrossEncoderBackend:
    """Cross-encoder scorer with an optional ONNX Runtime execution path.

    The ONNX graph is exported from the very same HuggingFace checkpoint and
    agrees with the torch path to ~5e-6 absolute, so this is an execution
    optimisation and *not* a model substitution: the SCM's R mechanism is
    unchanged. Falls back to torch if onnxruntime is unavailable.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int,
        use_onnx: bool = True,
        device: str = "cpu",
        fp16: bool = False,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.max_length = max_length
        self.device = device
        self._ce = CrossEncoder(model_name, device=device, max_length=max_length)
        self._ce.model.eval()
        if fp16 and device.startswith("cuda"):
            self._ce.model.half()
        self.tokenizer = self._ce.tokenizer
        self.backend = f"torch/{device}" + ("/fp16" if fp16 and device.startswith("cuda") else "")
        self._sess = None
        if use_onnx and not device.startswith("cuda"):
            try:
                self._sess = self._build_onnx(model_name)
                self.backend = "onnx"
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[pipeline] ONNX export failed ({exc}); using torch backend")

    def _build_onnx(self, model_name: str):
        import onnxruntime as ort
        import torch

        path = config.CACHE_DIR / f"ce_{model_name.split('/')[-1]}_{self.max_length}.onnx"
        if not path.exists():
            print("[pipeline] exporting cross-encoder to ONNX (one-off)")
            enc = self.tokenizer(
                ["a"], ["b"], padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            torch.onnx.export(
                self._ce.model,
                (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
                str(path),
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "b", 1: "s"},
                    "attention_mask": {0: "b", 1: "s"},
                    "token_type_ids": {0: "b", 1: "s"},
                    "logits": {0: "b"},
                },
                opset_version=14,
                do_constant_folding=True,
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = config.N_TORCH_THREADS
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> np.ndarray:
        if self._sess is None:
            return np.asarray(
                self._ce.predict(
                    pairs, batch_size=batch_size, show_progress_bar=False,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        out = []
        for i in range(0, len(pairs), batch_size):
            b = pairs[i : i + batch_size]
            e = self.tokenizer(
                [p[0] for p in b], [p[1] for p in b],
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors="np",
            )
            logits = self._sess.run(
                ["logits"],
                {
                    "input_ids": e["input_ids"].astype(np.int64),
                    "attention_mask": e["attention_mask"].astype(np.int64),
                    "token_type_ids": e["token_type_ids"].astype(np.int64),
                },
            )[0]
            out.append(logits.squeeze(-1).astype(np.float32))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
@dataclass
class Corpus:
    """The document collection plus everything derived from it that is
    *pre-treatment* (i.e. unaffected by any do() on the query)."""

    doc_ids: list[str]
    texts: list[str]
    titles: list[str]
    doc_index: dict[str, int]
    doc_len: np.ndarray  # (N,) token counts
    doc_content_tokens: list[frozenset[str]]
    ce_texts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ce_texts:
            # Pre-truncate to roughly CE_MAX_LENGTH wordpieces so the tokenizer
            # is not asked to process (and discard) the tail of long abstracts
            # on every one of the ~500k reranking calls. Word budget is
            # deliberately generous: truncation to the exact token limit still
            # happens inside the tokenizer, this only removes wasted work.
            budget = int(config.CE_MAX_LENGTH * 0.9)
            self.ce_texts = [" ".join(t.split()[:budget]) for t in self.texts]

    def __len__(self) -> int:
        return len(self.doc_ids)

    def idx(self, doc_id: str) -> int:
        return self.doc_index[doc_id]


@dataclass
class Queries:
    """Test queries and their graded relevance judgements."""

    query_ids: list[str]
    texts: dict[str, str]
    qrels: dict[str, dict[str, int]]  # qid -> {docid: relevance}

    def relevant(self, qid: str, min_rel: int = 1) -> set[str]:
        return {d for d, r in self.qrels.get(qid, {}).items() if r >= min_rel}


def load_corpus_and_queries() -> tuple[Corpus, Queries]:
    """Load NFCorpus from ir_datasets, with a pickle cache.

    Documents are ``title + " " + text``, matching standard BEIR practice.
    """
    if config.CORPUS_CACHE.exists():
        # Serialised as plain primitives, never as dataclass instances: a
        # pickled dataclass records its defining module, which is "__main__"
        # when this file is run via ``python -m pipeline`` and therefore
        # unloadable from any other module.
        with open(config.CORPUS_CACHE, "rb") as f:
            blob = pickle.load(f)
        corpus = Corpus(
            doc_ids=blob["doc_ids"],
            texts=blob["texts"],
            titles=blob.get("titles") or [""] * len(blob["doc_ids"]),
            doc_index={d: i for i, d in enumerate(blob["doc_ids"])},
            doc_len=blob["doc_len"],
            doc_content_tokens=[frozenset(s) for s in blob["doc_content_tokens"]],
        )
        queries = Queries(
            query_ids=blob["query_ids"], texts=blob["qtexts"], qrels=blob["qrels"]
        )
        return corpus, queries

    import ir_datasets

    ds = ir_datasets.load(config.DATASET)

    doc_ids: list[str] = []
    texts: list[str] = []
    titles: list[str] = []
    # BEIR doc schemas differ: nfcorpus has (doc_id, text, title, url), scifact
    # (doc_id, text, title), quora and fiqa have no title at all. Access
    # defensively so a schema difference is a no-op rather than an AttributeError.
    for d in ds.docs_iter():
        title = (getattr(d, "title", "") or "").strip()
        body = (getattr(d, "text", "") or "").strip()
        doc_ids.append(d.doc_id)
        titles.append(title)
        texts.append(f"{title} {body}".strip())

    doc_len = np.array([len(tokenize(t)) for t in texts], dtype=np.float32)
    doc_content = [frozenset(content_tokens(t)) for t in texts]
    corpus = Corpus(
        doc_ids=doc_ids,
        texts=texts,
        titles=titles,
        doc_index={d: i for i, d in enumerate(doc_ids)},
        doc_len=doc_len,
        doc_content_tokens=doc_content,
    )

    qtexts = {q.query_id: q.text for q in ds.queries_iter()}
    qrels: dict[str, dict[str, int]] = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = int(qr.relevance)

    # Keep only queries that have at least one judged-relevant document AND
    # appear in the qrels file; sort for determinism.
    qids = sorted(qid for qid in qtexts if qrels.get(qid))
    queries = Queries(query_ids=qids, texts=qtexts, qrels=qrels)

    with open(config.CORPUS_CACHE, "wb") as f:
        pickle.dump(
            {
                "doc_ids": doc_ids,
                "texts": texts,
                "titles": titles,
                "doc_len": doc_len,
                "doc_content_tokens": [set(s) for s in doc_content],
                "query_ids": qids,
                "qtexts": qtexts,
                "qrels": qrels,
            },
            f,
        )
    return corpus, queries


def select_queries(queries: Queries, n: int | None = None) -> list[str]:
    """Deterministic query subset. ``None`` -> all queries."""
    n = config.N_QUERIES if n is None else n
    if n is None or n >= len(queries.query_ids):
        return list(queries.query_ids)
    return list(queries.query_ids[:n])


# --------------------------------------------------------------------------- #
# Pipeline result
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    """Every intermediate value of one pipeline execution.

    Holding the intermediates explicitly is what lets :mod:`mediation` compare
    counterfactual worlds stage by stage.
    """

    bm25_query: str
    dense_query: str
    rerank_query: str
    first_stage: FirstStage
    bm25_full: np.ndarray  # (N,) raw BM25 scores over the whole corpus
    dense_full: np.ndarray  # (N,) cosine similarities over the whole corpus
    candidates: list[str]
    provenance: dict[str, str]  # docid -> bm25_only | dense_only | both
    reranked: list[tuple[str, float]]  # (docid, CE score), descending
    ranks: dict[str, int] = field(default_factory=dict)

    def rank_of(self, doc_id: str) -> int:
        """Truncated rank of ``doc_id`` in the reranked list.

        Documents outside the candidate pool - and those ranked below
        ``K_CANDIDATES`` - receive :data:`config.MISSING_RANK`. Rank is
        therefore a censored outcome on ``[1, K_CANDIDATES + 1]``.
        """
        return self.ranks.get(doc_id, config.MISSING_RANK)

    def top(self, k: int = config.K_FINAL) -> list[str]:
        return [d for d, _ in self.reranked[:k]]

    def ce_score_of(self, doc_id: str) -> float | None:
        for d, s in self.reranked:
            if d == doc_id:
                return s
        return None


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
class RetrievalPipeline:
    """Deterministic three-stage retrieval pipeline with independently
    overridable stages.

    Stage correspondence to the SCM:

    ``bm25_scores``  -> M1 (lexical mediator)
    ``dense_scores`` -> M2 (semantic mediator)
    ``candidates``   -> C  (collider: both mediators feed it)
    ``rerank``       -> R  (final mechanism)
    ``run``          -> Y  (the retrieved ranking)
    """

    def __init__(self, corpus: Corpus, verbose: bool = True) -> None:
        self.corpus = corpus
        self.verbose = verbose
        self._bm25 = None
        self._stemmer = None
        self._dense_model = None
        self._ce_model = None
        self._doc_emb: np.ndarray | None = None  # L2-normalised
        self._doc_emb_norm: np.ndarray | None = None  # pre-normalisation L2 norm
        self._ce_cache: dict[tuple[str, int], float] = {}
        self.ce_pairs_scored = 0
        self.ce_cache_hits = 0

    # ---------------------------- lazy resources --------------------------- #
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[pipeline] {msg}", flush=True)

    @property
    def bm25(self):
        if self._bm25 is None:
            self._build_bm25()
        return self._bm25

    def _build_bm25(self) -> None:
        import bm25s
        import Stemmer

        self._stemmer = Stemmer.Stemmer("english")
        idx_dir = config.BM25_INDEX_CACHE
        if idx_dir.exists():
            self._log("loading cached BM25 index")
            self._bm25 = bm25s.BM25.load(str(idx_dir), load_corpus=False)
            return
        self._log("building BM25 index")
        t0 = time.time()
        tokens = bm25s.tokenize(
            self.corpus.texts,
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
        retriever.index(tokens, show_progress=False)
        retriever.save(str(idx_dir))
        self._bm25 = retriever
        self._log(f"BM25 index built in {time.time() - t0:.1f}s")

    @property
    def stemmer(self):
        if self._stemmer is None:
            self._build_bm25()
        return self._stemmer

    @property
    def dense_model(self):
        if self._dense_model is None:
            from sentence_transformers import SentenceTransformer

            self._log(f"loading dense model {config.DENSE_MODEL} on {config.DEVICE}")
            self._dense_model = SentenceTransformer(config.DENSE_MODEL, device=config.DEVICE)
            self._dense_model.max_seq_length = config.MAX_SEQ_LENGTH
            if config.USE_FP16 and config.ON_GPU:
                self._dense_model.half()
        return self._dense_model

    @property
    def ce_model(self) -> CrossEncoderBackend:
        if self._ce_model is None:
            self._log(f"loading cross-encoder {config.CROSS_ENCODER_MODEL}")
            self._ce_model = CrossEncoderBackend(
                config.CROSS_ENCODER_MODEL,
                max_length=config.CE_MAX_LENGTH,
                use_onnx=config.USE_ONNX_CE,
                device=config.DEVICE,
                fp16=config.USE_FP16,
            )
            self._log(f"cross-encoder backend = {self._ce_model.backend}")
        return self._ce_model

    @property
    def doc_emb(self) -> np.ndarray:
        if self._doc_emb is None:
            self._load_embeddings()
        return self._doc_emb  # type: ignore[return-value]

    @property
    def doc_emb_norm(self) -> np.ndarray:
        """L2 norm of each document embedding *before* normalisation.

        Causal role: covariate X in the DML analysis. Embedding norm correlates
        with document length and topical specificity, and is not affected by
        the query - a legitimate pre-treatment confounder.
        """
        if self._doc_emb_norm is None:
            self._load_embeddings()
        return self._doc_emb_norm  # type: ignore[return-value]

    def _load_embeddings(self) -> None:
        if config.EMBEDDINGS_CACHE.exists():
            self._log("loading cached document embeddings")
            z = np.load(config.EMBEDDINGS_CACHE)
            self._doc_emb = z["emb"]
            self._doc_emb_norm = z["norm"]
            return
        self._log(f"embedding {len(self.corpus)} documents (one-off)")
        t0 = time.time()
        raw = self.dense_model.encode(
            self.corpus.texts,
            batch_size=config.EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=self.verbose,
        ).astype(np.float32)
        norm = np.linalg.norm(raw, axis=1).astype(np.float32)
        emb = raw / np.clip(norm[:, None], 1e-12, None)
        np.savez_compressed(config.EMBEDDINGS_CACHE, emb=emb, norm=norm)
        self._doc_emb, self._doc_emb_norm = emb, norm
        self._log(f"embedded in {time.time() - t0:.1f}s")

    # ------------------------------ stage M1 ------------------------------- #
    def _bm25_array(self, query: str) -> np.ndarray:
        """BM25 scores over the whole corpus (hot path, array-valued).

        Memoised per query string: the mediation runs re-issue the same baseline
        query dozens of times.
        """
        cached = self.__dict__.setdefault("_bm25_cache", {})
        hit = cached.get(query)
        if hit is not None:
            return hit
        import bm25s

        toks = bm25s.tokenize(
            query, stopwords="en", stemmer=self.stemmer, return_ids=False,
            show_progress=False,
        )[0]
        scores = (
            self.bm25.get_scores(toks)
            if toks
            else np.zeros(len(self.corpus), dtype=np.float32)
        )
        scores = np.asarray(scores, dtype=np.float32)
        if len(cached) > 8192:
            cached.clear()
        cached[query] = scores
        return scores

    def bm25_scores(self, query: str) -> dict[str, float]:
        """M1: lexical channel. Returns ``{doc_id: bm25 score}`` for all docs
        with a non-zero score."""
        arr = self._bm25_array(query)
        nz = np.nonzero(arr)[0]
        return {self.corpus.doc_ids[i]: float(arr[i]) for i in nz}

    # ------------------------------ stage M2 ------------------------------- #
    def _dense_array(self, query: str) -> np.ndarray:
        cached = self.__dict__.setdefault("_dense_cache", {})
        hit = cached.get(query)
        if hit is not None:
            return hit
        vec = self.dense_model.encode(
            [config.BGE_QUERY_PREFIX + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]
        scores = self.doc_emb @ vec
        if len(cached) > 8192:
            cached.clear()
        cached[query] = scores
        return scores

    def dense_scores(self, query: str) -> dict[str, float]:
        """M2: semantic channel. Cosine similarity over cached, L2-normalised
        document embeddings."""
        arr = self._dense_array(query)
        return {self.corpus.doc_ids[i]: float(arr[i]) for i in range(len(arr))}

    # ------------------------------- stage C ------------------------------- #
    def candidates(
        self,
        bm25_s: np.ndarray | dict[str, float],
        dense_s: np.ndarray | dict[str, float],
        first_stage: FirstStage = "union",
        k: int = config.K_CANDIDATES,
    ) -> tuple[list[str], dict[str, str]]:
        """C: the candidate pool, plus per-candidate channel provenance.

        Provenance ('bm25_only' / 'dense_only' / 'both') records *which* first
        stage put each document into the pool. It is the observable trace of
        which causal path was active for that document.
        """
        bm25_arr = self._as_array(bm25_s)
        dense_arr = self._as_array(dense_s)

        bm25_top = self._topk_ids(bm25_arr, k) if first_stage in ("union", "bm25") else []
        dense_top = self._topk_ids(dense_arr, k) if first_stage in ("union", "dense") else []

        bset, dset = set(bm25_top), set(dense_top)
        # Deterministic order: BM25 rank first, then dense-only in dense rank order.
        ordered = bm25_top + [d for d in dense_top if d not in bset]
        prov = {
            d: ("both" if d in bset and d in dset else "bm25_only" if d in bset else "dense_only")
            for d in ordered
        }
        return ordered, prov

    def _as_array(self, s: np.ndarray | dict[str, float]) -> np.ndarray:
        if isinstance(s, np.ndarray):
            return s
        arr = np.zeros(len(self.corpus), dtype=np.float32)
        for d, v in s.items():
            arr[self.corpus.idx(d)] = v
        return arr

    def _topk_ids(self, arr: np.ndarray, k: int) -> list[str]:
        k = min(k, len(arr))
        idx = np.argpartition(-arr, k - 1)[:k]
        # Stable tie-break on (-score, doc index) keeps the pool deterministic.
        idx = idx[np.lexsort((idx, -arr[idx]))]
        return [self.corpus.doc_ids[i] for i in idx]

    # ------------------------------- stage R ------------------------------- #
    def rerank(self, query: str, candidate_ids: list[str]) -> list[tuple[str, float]]:
        """R: cross-encoder rescoring of the candidate pool.

        Pair scores are memoised on ``(query, doc)``. The mediation experiment
        re-runs heavily overlapping (query, pool) combinations, so the cache
        turns an otherwise quadratic experiment into a near-linear one.
        """
        if not candidate_ids:
            return []
        cache = self._ce_cache
        need: list[str] = []
        for d in candidate_ids:
            if (query, self.corpus.idx(d)) not in cache:
                need.append(d)
            else:
                self.ce_cache_hits += 1

        if need:
            if len(cache) > config.CE_CACHE_MAX:
                cache.clear()
            pairs = [(query, self.corpus.ce_texts[self.corpus.idx(d)]) for d in need]
            scores = self.ce_model.predict(pairs, batch_size=config.CE_BATCH_SIZE)
            self.ce_pairs_scored += len(need)
            for d, s in zip(need, scores):
                cache[(query, self.corpus.idx(d))] = float(s)

        scored = [(d, cache[(query, self.corpus.idx(d))]) for d in candidate_ids]
        # Deterministic tie-break: descending score, then corpus order.
        scored.sort(key=lambda t: (-t[1], self.corpus.idx(t[0])))
        return scored

    # -------------------------------- run ---------------------------------- #
    def run(
        self,
        query: str,
        *,
        bm25_query: str | None = None,
        dense_query: str | None = None,
        rerank_query: str | None = None,
        first_stage: FirstStage = "union",
    ) -> PipelineResult:
        """Execute the pipeline, optionally feeding a *different* query to each
        stage.

        Passing distinct per-stage queries is exactly the "freezing" operation
        used by :mod:`mediation`: a stage fed ``Q0`` while the rest sees ``Q1``
        is held at its baseline value, so the observed change in the outcome is
        the effect transmitted through the *unfrozen* paths only.
        """
        bq = bm25_query if bm25_query is not None else query
        dq = dense_query if dense_query is not None else query
        rq = rerank_query if rerank_query is not None else query

        bm25_arr = (
            self._bm25_array(bq)
            if first_stage in ("union", "bm25")
            else np.zeros(len(self.corpus), dtype=np.float32)
        )
        dense_arr = (
            self._dense_array(dq)
            if first_stage in ("union", "dense")
            else np.zeros(len(self.corpus), dtype=np.float32)
        )
        cands, prov = self.candidates(bm25_arr, dense_arr, first_stage=first_stage)
        reranked = self.rerank(rq, cands)
        ranks = {
            d: i + 1
            for i, (d, _) in enumerate(reranked[: config.K_CANDIDATES])
        }
        return PipelineResult(
            bm25_query=bq,
            dense_query=dq,
            rerank_query=rq,
            first_stage=first_stage,
            bm25_full=bm25_arr,
            dense_full=dense_arr,
            candidates=cands,
            provenance=prov,
            reranked=reranked,
            ranks=ranks,
        )

    # --------------------- single-channel rankings (no CE) ----------------- #
    def rank_bm25_only(self, query: str, k: int = config.K_FINAL) -> list[str]:
        """Ranking produced by the lexical channel alone (no reranker)."""
        return self._topk_ids(self._bm25_array(query), k)

    def rank_dense_only(self, query: str, k: int = config.K_FINAL) -> list[str]:
        """Ranking produced by the semantic channel alone (no reranker)."""
        return self._topk_ids(self._dense_array(query), k)

    # ------------------------------ covariates ----------------------------- #
    def covariates(self, query: str, doc_id: str) -> dict[str, float]:
        """Per (query, doc) pre-treatment covariates used as confounders X."""
        i = self.corpus.idx(doc_id)
        qtok = content_tokens(query)
        return {
            "doc_len": float(self.corpus.doc_len[i]),
            "doc_emb_norm": float(self.doc_emb_norm[i]),
            "lex_overlap": jaccard(qtok, self.corpus.doc_content_tokens[i]),
            "query_len": float(len(tokenize(query))),
        }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def ndcg_at_k(ranking: list[str], rel: dict[str, int], k: int = config.K_FINAL) -> float:
    """Graded nDCG@k with the standard ``(2^rel - 1) / log2(i+1)`` gain."""
    dcg = sum(
        (2 ** rel.get(d, 0) - 1) / np.log2(i + 2) for i, d in enumerate(ranking[:k])
    )
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


@dataclass
class BaselineRun:
    """Cached baseline (no-intervention) execution for one query."""

    query_id: str
    query_text: str
    candidates: list[str]
    provenance: dict[str, str]
    reranked: list[tuple[str, float]]
    ranks: dict[str, int]
    bm25_top: list[str]
    dense_top: list[str]
    bm25_scores_cand: dict[str, float]
    dense_scores_cand: dict[str, float]


def compute_baseline(
    pipe: RetrievalPipeline, queries: Queries, qids: list[str]
) -> dict[str, BaselineRun]:
    """Run the un-intervened pipeline for every query and cache the result.

    Causal role: this is the factual world ``Q = q``. Every interventional and
    counterfactual quantity downstream is measured as a deviation from it.
    """
    cache_key = config.baseline_cache_path(qids)
    if cache_key.exists():
        with open(cache_key, "rb") as f:
            blob = pickle.load(f)
        runs = {k: BaselineRun(**v) for k, v in blob.items()}
        if set(runs) == set(qids):
            print(f"[pipeline] loaded cached baseline for {len(runs)} queries")
            return runs

    runs: dict[str, BaselineRun] = {}
    t0 = time.time()
    for n, qid in enumerate(qids, 1):
        qtext = queries.texts[qid]
        res = pipe.run(qtext)
        runs[qid] = BaselineRun(
            query_id=qid,
            query_text=qtext,
            candidates=res.candidates,
            provenance=res.provenance,
            reranked=res.reranked,
            ranks=res.ranks,
            bm25_top=pipe.rank_bm25_only(qtext, config.K_CANDIDATES),
            dense_top=pipe.rank_dense_only(qtext, config.K_CANDIDATES),
            bm25_scores_cand={d: float(res.bm25_full[pipe.corpus.idx(d)]) for d in res.candidates},
            dense_scores_cand={d: float(res.dense_full[pipe.corpus.idx(d)]) for d in res.candidates},
        )
        if n % 25 == 0 or n == len(qids):
            el = time.time() - t0
            print(
                f"[pipeline] baseline {n}/{len(qids)}  "
                f"{el:.0f}s  ({el / n:.2f}s/query)",
                flush=True,
            )
    with open(cache_key, "wb") as f:
        pickle.dump({k: vars(v) for k, v in runs.items()}, f)
    return runs


def baseline_sanity_check(
    pipe: RetrievalPipeline, queries: Queries, runs: dict[str, BaselineRun]
) -> dict[str, float]:
    """nDCG@10 for BM25-only, dense-only and the full pipeline.

    A full pipeline that fails to beat both single channels means the SCM is
    mis-wired; the experiment would be measuring an artefact.
    """
    n_b, n_d, n_f = [], [], []
    for qid, run in runs.items():
        rel = queries.qrels.get(qid, {})
        n_b.append(ndcg_at_k(run.bm25_top[: config.K_FINAL], rel))
        n_d.append(ndcg_at_k(run.dense_top[: config.K_FINAL], rel))
        n_f.append(ndcg_at_k([d for d, _ in run.reranked[: config.K_FINAL]], rel))

    out = {
        "bm25_only": float(np.mean(n_b)),
        "dense_only": float(np.mean(n_d)),
        "full_pipeline": float(np.mean(n_f)),
        "n_queries": len(runs),
        "dataset": config.DATASET,
    }
    full = out["full_pipeline"]
    best_single = max(out["bm25_only"], out["dense_only"])
    print("\n" + "=" * 62)
    print(f"BASELINE SANITY CHECK - nDCG@10 on {config.DATASET}")
    print("=" * 62)
    print(f"  BM25 only      : {out['bm25_only']:.4f}")
    print(f"  Dense only     : {out['dense_only']:.4f}")
    print(f"  Full pipeline  : {full:.4f}")
    print(f"  queries        : {out['n_queries']}")

    # Recorded ranges describe the FULL query set. Applying them to a smoke run
    # produces false alarms - a 10-query sample of NFCorpus lands at 0.31 purely
    # by sampling - which trains you to ignore the one warning that matters.
    expected = config.BASELINE_EXPECTATIONS.get(config.DATASET)
    if expected is not None and config.N_QUERIES is not None:
        print(
            f"  (subset run, n={config.N_QUERIES}: recorded range "
            f"{list(expected)} not applied - it describes the full query set)"
        )
        expected = None
        out["expectation"] = None
    elif expected is None:
        print(
            f"  NOTE: no recorded expectation for {config.DATASET}. Applying the "
            f"universal floor ({config.BASELINE_FLOOR}) only; record "
            f"{full:.4f} in config.BASELINE_EXPECTATIONS after inspecting this run."
        )
        out["expectation"] = None
    else:
        lo, hi = expected
        out["expectation"] = [lo, hi]
        if not (lo <= full <= hi):
            print(
                f"  WARNING: full-pipeline nDCG@10 {full:.4f} is outside the "
                f"recorded range [{lo}, {hi}] for this dataset. A quality "
                "regression invalidates every downstream causal number."
            )
        else:
            print(f"  within recorded range [{lo}, {hi}]")

    out["reranker_helps"] = bool(full > best_single)
    if not out["reranker_helps"]:
        known = config.DATASET in config.KNOWN_RERANKER_HARMFUL
        print(
            f"  WARNING: full pipeline ({full:.4f}) does NOT beat the best single "
            f"channel ({best_single:.4f}) - the reranker DEGRADES this collection."
            + ("\n           This is a known property of this dataset, not a bug: "
               "ms-marco-MiniLM is trained on web-style queries." if known else
               "\n           Check for a mis-wired reranker before trusting anything downstream.")
            + "\n           Mediation shares below still describe causal responsibility "
              "for the intervention's effect, but NOT a well-configured pipeline."
        )
    if full < config.BASELINE_FLOOR:
        print(
            f"  WARNING: full-pipeline nDCG@10 < {config.BASELINE_FLOOR}. "
            "Investigate before running modules 2-5."
        )
    print("=" * 62 + "\n")
    return out


def main() -> None:
    config.set_seeds()
    t0 = time.time()
    print(f"[pipeline] {config.device_banner()}")
    corpus, queries = load_corpus_and_queries()
    print(f"[pipeline] corpus={len(corpus)} docs, queries={len(queries.query_ids)}")
    pipe = RetrievalPipeline(corpus)
    qids = select_queries(queries)
    print(f"[pipeline] running baseline on {len(qids)} queries")
    runs = compute_baseline(pipe, queries, qids)
    baseline_sanity_check(pipe, queries, runs)
    print(f"[pipeline] total wall clock {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
