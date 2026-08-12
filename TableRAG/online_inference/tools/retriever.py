import os
import time
import urllib3
import sys
import json
import transformers
import warnings
import threading
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModel
import transformers
from utils.tool_utils import *
import nltk
from more_itertools import chunked
import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional
from utils.utils import read_plain_csv

import lancedb

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


class SemanticRetriever :
    """
    Retrieving process, containing recall and rerank.
    Vector store: LanceDB (embedded, no server, persistent on disk).
    """
    def __init__(
        self,
        chunks: List[str],
        chunk_index: Dict,
        chunk_file_index: Optional[Dict] = None,
        llm_path: Optional[str] = None,
        reranker_path: Optional[str] = None,
        save_path: str = "./retrieval_result/lancedb"
    ) -> None:
        self.embedding_model = Embedder(llm_path)
        self.reranker = Reranker(reranker_path)

        self.chunks = chunks
        self.chunk_index = chunk_index
        self.chunk_file_index = chunk_file_index or {}

        # LanceDB persistent storage (save_path = direktori).
        db_path = os.path.dirname(save_path) or "."
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self.table_name = "doc_chunks"

        if self.table_name in self.db.list_tables() :
            self.table = self.db.open_table(self.table_name)
        else :
            self.table = self._build_table(chunks)

        self.thread_local = threading.local()
        self.index_lock = threading.RLock()

    def _build_table(self, chunks: List[str]) -> Any :
        """Embed semua chunk lalu simpan sebagai tabel LanceDB."""
        print("Building LanceDB table ...")
        embeddings = self.embed_doc(chunks)
        data = [
            {
                "id": str(i),
                "vector": embeddings[i].tolist(),
                "text": chunks[i],
                "filename": self.chunk_file_index.get(i, ""),
            }
            for i in range(len(chunks))
        ]
        return self.db.create_table(self.table_name, data=data)

    def embed_doc(self, chunks: List[str], batch_size: int = 512) -> Any :
        """
        Embed documents in batches for improved performance

        Args:
            chunks: List of text chunks to embed

        Returns:
            np.ndarray: Array of document embeddings
        """
        encode_vecs = []
        iterator = tqdm(range(0, len(chunks), batch_size)) if len(chunks) >= 100 \
            else range(0, len(chunks), batch_size)

        for i in iterator :
            batch = chunks[i: i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)

            if len(batch) == 1 :
                encode_vecs.append(batch_embeddings)
            else :
                encode_vecs.extend(batch_embeddings)

        encode_vecs = np.array(encode_vecs)
        if len(encode_vecs.shape) == 3 :
            encode_vecs = encode_vecs.reshape(-1, encode_vecs.shape[-1])

        return encode_vecs

    def retrieve(self, query, recall_num, rerank_num) :
        docs, ori_file_name = self.recall(query, recall_num)
        reranked_docs, rerank_scores, filenames = self.rerank(query, docs, rerank_num, ori_file_name)
        return reranked_docs, rerank_scores, filenames

    def recall(self, query: str, topn: int) -> Tuple[List[str], List[str]] :
        query_emb = self.embed_doc([query])[0].tolist()
        with self.index_lock :
            rows = self.table.search(query_emb).limit(topn).to_list()
        ori_docs = [row["text"] for row in rows]
        ori_file_name = [row["filename"] for row in rows]
        return ori_docs, ori_file_name

    def rerank(self, query: str, docs: List[str], topn: int, ori_file_name: List[str]) -> Tuple[List[str], List[float], List[str]] :
        pairs = [[query, d] for d in docs]
        scores = self.reranker.compute_score(pairs)
        sroted_pairs = sorted(zip(scores, docs, ori_file_name), reverse=True)
        score_sorted, doc_sorted, filename_sorted = zip(*sroted_pairs)
        return list(doc_sorted)[:topn], list(score_sorted)[:topn], list(filename_sorted)[:topn]


class MixedDocRetriever :
    def __init__(
        self,
        doc_dir_path: str,
        excel_dir_path: str,
        llm_path: Optional[str] = None,
        reranker_path: Optional[str] = None,
        save_path: str = "./retrieval_result/lancedb"
    ) -> None:
        self.ori_documents = self.load_hybrid_dataset(doc_dir_path, excel_dir_path)
        print("Loading done.")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_chunking_dict = self.doc_chunking()
        self.chunks, self.chunk_to_index, self.chunk_to_filename = self.build_index(doc_chunking_dict)
        self.semantic_retriever = SemanticRetriever(
            chunks=self.chunks,
            chunk_index=self.chunk_to_index,
            chunk_file_index=self.chunk_to_filename,
            llm_path=llm_path,
            reranker_path=reranker_path,
            save_path=save_path
        )

    def load_hybrid_dataset(self, doc_dir_path: str, excel_dir_path: str) -> Dict[str, str] :
        all_docs: Dict[str, str] = {}
        for file in tqdm(os.listdir(excel_dir_path)) :
            content = excel_to_markdown(os.path.join(excel_dir_path, file))
            excel_content = content
            all_docs[file] = excel_content

        for file in tqdm(os.listdir(doc_dir_path)) :
            with open(os.path.join(doc_dir_path, file), 'r', encoding="utf-8") as fin :
                data_split = json.load(fin)
            key_value_doc = ''
            for key, item in data_split.items() :
                key_value_doc += f"{key} {item}\n"

            all_docs[file] = key_value_doc
        return all_docs

    def build_index(self, chunking_dict: Dict) -> Tuple :
        flatten_chunks = []
        chunk_to_index = defaultdict(list)
        chunk_to_filename = defaultdict(str)
        cnt = 0
        for idx, (key, item) in enumerate(chunking_dict.items()) :
            flatten_chunks += item
            for i in item :
                chunk_to_index[cnt] = i
                chunk_to_filename[cnt] = key
                cnt += 1
        return flatten_chunks, chunk_to_index, chunk_to_filename

    def nltk_single_doc_chunking(self, doc: str, key: str) -> Tuple[List[str], str] :
        all_splits = self.text_splitter.split_text(doc)
        add_file_name_splits = []
        for split_chunk in all_splits :
            add_file_name_chunk = f"File name: {key}\n" + split_chunk
            add_file_name_splits.append(add_file_name_chunk)
        return add_file_name_splits, key

    def doc_chunking(self, max_workers=10) -> Dict :
        doc_chunkings = defaultdict(list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor :
            future_to_case = {executor.submit(self.nltk_single_doc_chunking, doc, key): doc for key, doc in self.ori_documents.items()}
            for future in tqdm(as_completed(future_to_case), total=len(self.ori_documents), desc="Processing cases") :
                try :
                    single_doc_chunking, key = future.result()
                    doc_chunkings[key] += single_doc_chunking
                except Exception as e :
                    print(f"Case processing generated exception: {e}")
        return doc_chunkings

    def retrieve(self, query: str, recall_nun: int = 50, rerank_num: int = 5) :
        return self.semantic_retriever.retrieve(query, recall_nun, rerank_num)
