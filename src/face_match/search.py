# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict, cast

import cv2
import faiss
import numpy as np

from face_match.core import (
    CACHE_NAME,
    SFACE_NAME,
    YUNET_NAME,
    embed,
    ensure_model,
    list_image_paths,
    load_bgr,
    load_cache,
    save_cache,
)


class SearchResult(TypedDict):
    distance: float
    path: Path
    metric: str


class SearchResponse(TypedDict):
    query: Path
    db: Path
    total_scanned: int
    with_face: int
    results: list[SearchResult]
    metric: str
    threshold: float
    engine: str
    device: str


def find_matches(
    query: Path,
    db: Path,
    top: int = 10,
    distance: int = 0,
    rebuild_cache: bool = False,
    threshold: float | None = None,
    device: str = "cpu",
) -> SearchResponse:
    """
    Search for the most similar faces in a database folder.

    Args:
        query: Path to the query image.
        db: Path to the gallery/database folder.
        top: Maximum number of results to return.
        distance: 0 for Cosine similarity, 1 for L2 distance.
        rebuild_cache: If True, ignores existing JSON cache and rescans all images.
        threshold: Similarity threshold (optional, uses defaults if None).
        device: Execution device, "cpu" or "gpu".

    Returns:
        A dictionary containing metadata and a list of sorted results.
    """
    if threshold is None:
        threshold = 0.363 if distance == 0 else 1.128

    images = list_image_paths(db)
    det_path = ensure_model(YUNET_NAME)
    rec_path = ensure_model(SFACE_NAME)

    q_img = load_bgr(query)
    if q_img is None:
        raise ValueError(f"Could not read the query image: {query.name}")
        
    h0, w0 = q_img.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(det_path), "", (w0, h0), 0.9, 0.3, 5000, 0, 0)
    recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")

    if device == "gpu":
        detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # type: ignore[attr-defined]
        detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # type: ignore[attr-defined]
        recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # type: ignore[attr-defined]
        recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # type: ignore[attr-defined]

    q_feat = embed(q_img, detector, recognizer)
    if q_feat is None:
        raise ValueError("No face detected in the query image.")

    cache_path = db / CACHE_NAME
    cache: dict[str, tuple[float, np.ndarray]] = load_cache(cache_path) if not rebuild_cache else {}
    
    all_feats = []
    all_paths = []
    need_save = rebuild_cache

    for imp in images:
        # Use relative path for cache (Portability & Privacy)
        try:
            rel_path = imp.relative_to(db)
            key = str(rel_path.as_posix())
        except ValueError:
            # If not relative for some reason, use the filename
            key = imp.name
        
        if imp.resolve() == query.resolve():
            continue
        
        mtime = imp.stat().st_mtime
        entry = cache.get(key)
        feat: np.ndarray | None = None
        
        if (
            not rebuild_cache
            and isinstance(entry, tuple)
            and len(entry) == 2
            and entry[0] == mtime
        ):
            feat = np.asarray(entry[1])
        else:
            bgr = load_bgr(imp)
            if bgr is None:
                continue
            feat = embed(bgr, detector, recognizer)
            if feat is None:
                continue
            cache[key] = (mtime, feat)
            need_save = True
        
        all_feats.append(feat.flatten())
        all_paths.append(imp)

    if need_save and cache:
        save_cache(cache_path, cache)

    if not all_feats:
        return cast(SearchResponse, {
            "query": query, "db": db, "total_scanned": len(images),
            "with_face": 0, "results": [], "metric": "coseno" if distance == 0 else "L2",
            "threshold": threshold, "engine": "FAISS", "device": "CPU"
        })

    xb = np.array(all_feats).astype("float32")
    xq: np.ndarray = q_feat.reshape(1, -1).astype("float32")
    
    if distance == 0:
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
        index = faiss.IndexFlatIP(xb.shape[1])
        metric_name = "coseno"
    else:
        index = faiss.IndexFlatL2(xb.shape[1])
        metric_name = "L2"

    index.add(xb)
    
    used_device = "CPU"
    if device == "gpu":
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            used_device = "GPU"
        except Exception:
            used_device = "CPU"
    
    D, indices = index.search(xq, len(all_paths))
    
    results = []
    for dist_val, idx in zip(D[0], indices[0]):
        if idx == -1:
            continue
        d = float(dist_val)
        if distance == 0:  # Coseno
            if d < threshold:
                continue
        else:  # L2
            if d > threshold:
                continue
            
        results.append({
            "distance": d,
            "path": all_paths[idx],
            "metric": metric_name
        })

    results.sort(key=lambda x: x["distance"], reverse=(distance == 0))

    return cast(SearchResponse, {
        "query": query,
        "db": db,
        "total_scanned": len(images),
        "with_face": len(all_paths),
        "results": results[:top],
        "metric": metric_name,
        "threshold": threshold,
        "engine": "FAISS",
        "device": used_device
    })


def run_search(
    query: Path,
    db: Path,
    top: int,
    distance: int,
    rebuild_cache: bool,
    threshold: float,
    device: str = "cpu",
) -> int:
    try:
        data = find_matches(
            query=query, db=db, top=top, distance=distance,
            rebuild_cache=rebuild_cache, threshold=threshold, device=device
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    results = data["results"]
    metric_name = data["metric"]
    
    print()
    print(f"Query: {data['query'].name}")
    print(
        f"Database: {data['db'].name}  "
        f"({data['total_scanned']} images scanned, "
        f"{data['with_face']} with faces detected, "
        f"{len(results)} results returned)"
    )
    
    engine = data.get("engine", "FAISS")
    device = data.get("device", "CPU")
    desc = "higher values = more similar" if metric_name == "coseno" else "lower values = more similar"
    
    print(f"Metric: {metric_name} ({desc}) [Engine: {engine} {device}]")
    print()

    if not results:
        print(
            f"No matches found. None of the images exceeded the similarity threshold ({data['threshold']:.3f}).",
            file=sys.stderr,
        )
        return 1

    for i, res in enumerate(results):
        # Show relative path for privacy
        try:
            display_path = str(res["path"].relative_to(data["db"]))
        except ValueError:
            display_path = res["path"].name
            
        print(f"{i + 1}.  dist={res['distance']:.4f}  {res['metric']}  {display_path}")

    return 0


