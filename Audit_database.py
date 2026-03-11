"""
Full audit tool: Missing abstracts + Corrupted metadata + Deletion.
(Version with registry to prevent re-indexing of manually modified files)
"""
import os
import json
import warnings
import pathlib
warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import refchat_config as cfg

# File to store names of manually modified PDFs
AUDIT_LOG_FILE = cfg.PERSONAL_DATA / "audit_modifications.json"

def load_audit_log():
    if AUDIT_LOG_FILE.exists():
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_audit_log(modified_files):
    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(list(modified_files), f, indent=4)

def main():
    print("⏳ Loading database...")

    embedding_func = HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    db_path = cfg.get("db_path", "chroma_db")
    db = Chroma(persist_directory=db_path, embedding_function=embedding_func)

    # Load the existing registry
    already_audited = load_audit_log()

    print("🔍 Scanning the entire database...")
    total_chunks = db._collection.count()
    print(f"   (Analysing {total_chunks} text blocks...)")

    metadatas = []
    for offset in range(0, total_chunks, 2000):
        results = db._collection.get(limit=2000, offset=offset, include=["metadatas"])
        if results and "metadatas" in results:
            metadatas.extend(results["metadatas"])

    all_files            = {}
    files_with_abstract  = set()
    files_bad_meta       = set()

    for m in metadatas:
        if not m: continue
        fname = m.get("filename")
        if not fname: continue

        if fname not in all_files:
            all_files[fname] = m

        if m.get("section") == "Abstract":
            files_with_abstract.add(fname)

        if not m.get("auteur") or not m.get("annee"):
            files_bad_meta.add(fname)

    files_no_abstract = set(all_files.keys()) - files_with_abstract
    files_to_review   = files_no_abstract.union(files_bad_meta)
    total_to_review   = len(files_to_review)

    if total_to_review == 0:
        print("\n🎉 GREAT NEWS: Your database is perfect!")
        return

    print(f"\n⚠️  WARNING: {total_to_review} file(s) require your attention.\n")

    newly_modified = set()

    for idx, fname in enumerate(sorted(files_to_review), 1):
        print("\n" + "═" * 75)
        print(f"📄 File     : {fname}")
        print(f"📊 Progress : [ {idx} / {total_to_review} ] files processed")

        reasons = []
        if fname in files_bad_meta: reasons.append("Author or Year missing")
        if fname in files_no_abstract: reasons.append("Abstract missing")
        print(f"⚠️  Issue    : {' | '.join(reasons)}")
        print("─" * 75)

        base_meta = all_files[fname].copy()

        current_author = base_meta.get('auteur') or ""
        current_year   = base_meta.get('annee') or ""
        current_title  = base_meta.get('titre') or ""
        current_fname  = base_meta.get('filename') or fname

        print(f"🏷️  Current metadata: {current_author} ({current_year}) - {current_title[:40]}...")

        print("\n👉 What would you like to do with this file?")
        print("   [c] Correct metadata / file name")
        print("   [s] PERMANENTLY delete this file from the AI database")
        print("   [n] Do nothing (skip to abstract addition)")
        print("   [q] Quit the audit")

        choice = input("\n   Your choice (c/s/n/q): ").strip().lower()

        if choice == 'q':
            break

        elif choice == 's':
            confirm = input("   ⚠️ REALLY DELETE all excerpts from this file? (y/n): ").strip().lower()
            if confirm == 'y':
                print("   🗑️ Deleting...")
                existing = db._collection.get(where={"filename": fname})
                if existing and existing["ids"]:
                    db._collection.delete(ids=existing["ids"])
                    print(f"   ✅ {len(existing['ids'])} excerpts successfully deleted!")
                    newly_modified.add(fname)
                continue

        elif choice == 'c':
            n_fname  = input(f"   New File Name    [{current_fname}]: ").strip()
            n_author = input(f"   New Author       [{current_author}]: ").strip()
            n_year   = input(f"   New Year         [{current_year}]: ").strip()
            n_title  = input(f"   New Title        [{current_title[:30]}...]: ").strip()

            if n_fname or n_author or n_year or n_title:
                print("   🔄 Updating the entire database for this article...")
                existing = db._collection.get(where={"filename": fname}, include=["metadatas"])
                if existing and existing["ids"]:
                    ids_to_update   = existing["ids"]
                    metas_to_update = existing["metadatas"]

                    for m_up in metas_to_update:
                        if n_fname:  m_up["filename"] = n_fname
                        if n_author: m_up["auteur"]   = n_author
                        if n_year:   m_up["annee"]    = n_year
                        if n_title:  m_up["titre"]    = n_title

                    db._collection.update(ids=ids_to_update, metadatas=metas_to_update)
                    print(f"   ✅ {len(ids_to_update)} excerpts repaired!")
                    newly_modified.add(fname)
                    if n_fname: fname = n_fname

        if fname in files_no_abstract or base_meta.get('filename') in files_no_abstract:
            abs_choice = input("\n👉 Would you like to paste the abstract now? (y/n/q to quit): ").strip().lower()

            if abs_choice == 'q':
                break
            elif abs_choice == 'y':
                print("\n📝 Paste the abstract text below.")
                print("   (Press Enter, type 'END', then press Enter again to confirm):")

                lines = []
                while True:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    lines.append(line)

                abstract_text = "\n".join(lines).strip()

                if len(abstract_text) > 20:
                    print("\n⚙️ Integrating...")
                    is_e5 = "e5" in cfg.EMBEDDING_MODEL.lower()
                    text_to_embed = f"passage: {abstract_text}" if is_e5 else abstract_text

                    new_meta = base_meta.copy()
                    new_meta["section"]     = "Abstract"
                    new_meta["source_ajout"] = "manual"

                    doc = Document(page_content=text_to_embed, metadata=new_meta)
                    db.add_documents([doc])
                    newly_modified.add(fname)
                    print("✅ Abstract successfully injected into the database!")
                else:
                    print("❌ Text too short, cancelled.")
            else:
                print("⏭️ Abstract skipped for now.")

    # Final save of the registry
    if newly_modified:
        already_audited.update(newly_modified)
        save_audit_log(already_audited)
        print(f"\n💾 Audit registry updated ({len(newly_modified)} modifications saved).")

if __name__ == "__main__":
    main()
