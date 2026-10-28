"""仅 Protect AI 分类器；批量扫描两个视图，保留逐窗口分数。"""
import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

from common import (ROOT, DEFAULT_PACKAGE, read_json, write_json, load_materials, sha,
                    file_sha, explicit_text, external_view, split_sections,
                    append_record, verify_frozen, utcnow)


def window_starts(length, window, stride):
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    plan_hash = verify_frozen()
    config = read_json(ROOT / "configs/experiment.json")
    lock = read_json(ROOT / "configs/snapshot_lock.json")["detector"]
    receipt = read_json(ROOT / "runtime/download_detector.json")
    assert receipt["revision"] == lock["revision"] and receipt["repository_id"] == lock["repo"]
    assert receipt["files_sha256"] and any(x.endswith(".safetensors") for x in receipt["files_sha256"])
    model_path = Path(receipt["path"])
    for name, digest in receipt["files_sha256"].items():
        assert file_sha(model_path / name) == digest
    assert file_sha(args.package / config["source_materials_filename"]) == config["source_materials_sha256"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True).eval()
    # 官方标签0=良性、1=注入；核验类别数量并保存实际标签映射。
    assert model.config.num_labels == 2
    assert {int(k): str(v).upper() for k, v in model.config.id2label.items()} == {
        0: "SAFE", 1: "INJECTION"
    }, "检测器标签映射与冻结解释不一致，暂停而不是猜测"
    dcfg = config["detector"]
    assert dcfg["window_tokens"] + tokenizer.num_special_tokens_to_add(False) <= 512
    probe = tokenizer.encode("Tokenizer structure validation.", add_special_tokens=False)
    assert tokenizer.encode("Tokenizer structure validation.", add_special_tokens=True) == [
        tokenizer.cls_token_id, *probe, tokenizer.sep_token_id
    ], "单序列特殊符号结构与固定DeBERTa快照不一致"
    assert tokenizer.num_special_tokens_to_add(False) == 2
    rows, _ = load_materials(args.package)
    refs = [json.loads(line) for line in (ROOT / "materials/material_references.jsonl").read_text(
        encoding="utf-8").splitlines()]
    out_path = ROOT / "outputs/detector_records.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert record["plan_sha256"] == plan_hash
            key = (record["material_key"], record["view"])
            assert key not in done
            done.add(key)
    started = time.perf_counter()
    new_records = 0
    for ref in refs:
        condition = ref["input_condition"]
        emotion = ref.get("source_emotion") or ref["emotion"]
        source = rows[(ref["scenario"], emotion)]
        text = (explicit_text(source["neutral"]["full_user_message"]) if condition == "explicit_positive"
                else source[condition]["full_user_message"])
        assert sha(text) == ref["prompt_sha256"]
        for view in dcfg["views"]:
            if (ref["material_key"], view) in done:
                continue
            content = text if view == "full" else external_view(text)
            assert sha(content) == ref["prompt_sha256" if view == "full" else "external_sha256"]
            enc = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = enc["input_ids"], enc["offset_mapping"]
            starts = window_starts(len(ids), dcfg["window_tokens"], dcfg["stride_tokens"])
            windows = []
            sections = split_sections(content)
            for offset in range(0, len(starts), dcfg["batch_size"]):
                chunk_starts = starts[offset:offset + dcfg["batch_size"]]
                prepared = []
                for s in chunk_starts:
                    window_ids = [tokenizer.cls_token_id,
                                  *ids[s:s + dcfg["window_tokens"]],
                                  tokenizer.sep_token_id]
                    prepared.append({"input_ids": window_ids, "attention_mask": [1] * len(window_ids)})
                batch = tokenizer.pad(prepared, padding=True, return_tensors="pt")
                with torch.inference_mode():
                    scores = model(**batch).logits.softmax(dim=-1)[:, 1].cpu().tolist()
                for start, score in zip(chunk_starts, scores):
                    end = min(len(ids), start + dcfg["window_tokens"])
                    char_start, char_end = offsets[start][0], offsets[end - 1][1]
                    labels = [label for label, a, b in sections if a < char_end and b > char_start]
                    windows.append({"token_start": start, "token_end": end,
                                    "char_start": char_start, "char_end": char_end,
                                    "segment_labels": labels, "score": score})
            best = max(range(len(windows)), key=lambda i: windows[i]["score"])
            score = windows[best]["score"]
            append_record(out_path, {
                **ref, "view": view, "detector": lock["repo"], "detector_revision": lock["revision"],
                "score": score, "detected": score >= dcfg["threshold"], "threshold": dcfg["threshold"],
                "token_count": len(ids), "window_count": len(windows),
                "max_window_index": best, "windows": windows,
                "plan_sha256": plan_hash, "generated_at": utcnow(),
            })
            done.add((ref["material_key"], view))
            new_records += 1
            if new_records % 40 == 0:
                print(f"检测已完成 {len(done)}/1360 条视图记录，耗时 {time.perf_counter() - started:.1f} 秒", flush=True)
    write_json(ROOT / "outputs/detector_environment.json", {
        "completed_at": utcnow(), "plan_sha256": plan_hash, "id2label": model.config.id2label,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "huggingface-hub")},
        "threads": args.threads, "source_code_sha256": file_sha(Path(__file__)),
        "record_count": len(done), "elapsed_seconds_this_session": time.perf_counter() - started,
        "limitation": "检测结论仅基于单一检测器",
    })
    print("Protect AI 扫描完成。")


if __name__ == "__main__":
    main()
