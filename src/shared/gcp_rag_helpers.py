import json
import os

from google.cloud import storage
import vertexai
from vertexai import rag


def upload_to_gcs(local_file: str, bucket: str, dest_blob: str) -> str:
    """
    Upload a file to Google Cloud Storage.

    Args:
        local_file: Path to the local file to upload
        bucket: Name of the GCS bucket
        dest_blob: Destination path/name in the bucket

    Returns:
        GCS URI (gs://bucket/path)
    """
    storage_client = storage.Client()
    bucket_name = bucket.replace("gs://", "")
    blob = storage_client.bucket(bucket_name).blob(dest_blob)
    blob.upload_from_filename(local_file)
    gcs_uri = f"gs://{bucket_name}/{dest_blob}"
    print(f"Uploaded: {gcs_uri}")
    return gcs_uri


def import_files_to_rag_corpus(
    corpus_name: str,
    gcs_uris: list[str],
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
) -> dict:
    """
    Import files from GCS into a Vertex AI RAG corpus.
    Uses GCP_PROJECT_ID and GCP_LOCATION from environment variables.

    Args:
        corpus_name: Full corpus name (projects/{project}/locations/{location}/ragCorpora/{id})
        gcs_uris: List of GCS URIs to import (gs://bucket/path)
        chunk_size: Size of each text chunk (default: 1024)
        chunk_overlap: Overlap between chunks (default: 256)

    Returns:
        Dictionary with import results
    """
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION")

    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required")
    if not location:
        raise ValueError("GCP_LOCATION environment variable is required")

    vertexai.init(project=project_id, location=location)
    resp = rag.import_files(
        corpus_name=corpus_name,
        paths=gcs_uris,
        transformation_config=rag.TransformationConfig(
            rag.ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ),
    )

    result = {
        "imported_count": resp.imported_rag_files_count,
        "skipped_count": resp.skipped_rag_files_count,
    }
    print(f"Imported: {result['imported_count']}, Skipped: {result['skipped_count']}")
    return result


def list_corpus_files(corpus_name: str) -> list:
    """
    List all files in the RAG corpus with metadata.

    Args:
        corpus_name: Full corpus name (projects/{project}/locations/{location}/ragCorpora/{id})

    Returns:
        List of RagFile objects with metadata like display_name, size_bytes, rag_file_chunks_count
    """
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION")

    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required")
    if not location:
        raise ValueError("GCP_LOCATION environment variable is required")

    vertexai.init(project=project_id, location=location)
    files = rag.list_files(corpus_name=corpus_name)
    return list(files)


def upload_file_by_index(
    index: int,
    bucket: str,
    corpus_name: str,
    files_dir: str = "files_to_upload",
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
) -> dict:
    """
    Upload a file from files_to_upload directory by index and import to RAG corpus.

    Args:
        index: Index of the file to upload (0, 1, 2, etc.)
        bucket: Name of the GCS bucket
        corpus_name: Full corpus name (projects/{project}/locations/{location}/ragCorpora/{id})
        files_dir: Directory containing files to upload (default: "files_to_upload")
        chunk_size: Size of each text chunk (default: 1024)
        chunk_overlap: Overlap between chunks (default: 256)

    Returns:
        Dictionary with file name, GCS URI, and import results
    """
    files = sorted(os.listdir(files_dir))
    if index < 0 or index >= len(files):
        raise IndexError(f"Index {index} out of range. Found {len(files)} files.")

    filename = files[index]
    print(f"Uploading file: {filename}")
    local_path = os.path.join(files_dir, filename)

    gcs_uri = upload_to_gcs(local_path, bucket, filename)
    import_result = import_files_to_rag_corpus(
        corpus_name=corpus_name,
        gcs_uris=[gcs_uri],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return {
        "filename": filename,
        "gcs_uri": gcs_uri,
        **import_result,
    }


def upload_file_with_metadata(
    corpus_name: str,
    file_path: str,
    display_name: str = "",
    description: str = "",
    user_metadata: dict | None = None,
) -> dict:
    """
    Upload a file directly to a RAG corpus with metadata (display name, description, and custom user metadata).
    Uses the rag.upload_file() API which supports metadata, unlike import_files().

    Args:
        corpus_name: Full corpus name (projects/{project}/locations/{location}/ragCorpora/{id})
        file_path: Path to the local file to upload
        display_name: Display name for the file in the corpus (default: filename)
        description: Description for the file in the corpus (default: "")
        user_metadata: Dictionary of custom metadata (e.g., {"page_numbers": "1-50", "source": "2023 Q4 Report"})
                      This is stored as JSON and can be used for filtering/search

    Returns:
        Dictionary with upload results and metadata
    """
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION")

    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required")
    if not location:
        raise ValueError("GCP_LOCATION environment variable is required")

    # Use filename as display_name if not provided
    if not display_name:
        display_name = os.path.basename(file_path)

    vertexai.init(project=project_id, location=location)

    print(f"Uploading file with metadata: {display_name}")

    # Convert user_metadata dict to JSON string if provided
    metadata_json = None
    if user_metadata:
        metadata_json = json.dumps(user_metadata)
        print(f"  Custom metadata: {metadata_json}")

    rag_file = rag.upload_file(
        corpus_name=corpus_name,
        path=file_path,
        display_name=display_name,
        description=description,
        user_metadata=metadata_json,
    )

    result = {
        "name": rag_file.display_name,
        "description": rag_file.description if rag_file.description else "",
        "user_metadata": rag_file.user_metadata if rag_file.user_metadata else "",
        "file_resource": str(rag_file.name),
    }
    print(f"Uploaded with metadata: {result}")
    return result
