import os
from pathlib import Path
from typing import Optional, Union
import zipfile

from huggingface_hub import hf_hub_download


def _safe_extract_member(
    zip_file: zipfile.ZipFile, member: str, target: str, out_dir: Path
) -> None:
    target_path = (out_dir / target).resolve()
    if os.path.commonpath([str(out_dir), str(target_path)]) != str(out_dir):
        raise ValueError(
            f"Refusing to extract archive member outside target directory: {member}"
        )
    if member.endswith("/"):
        target_path.mkdir(parents=True, exist_ok=True)
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with zip_file.open(member) as src, target_path.open("wb") as dst:
        dst.write(src.read())


_REPO_ID = "simon-donike/SR-GAN"
_DATASET_FILENAME = "data/examples/dataset.zip"
# This commit contains the structured model and sample-data layout.  It is
# deliberately shared with inference presets so examples remain reproducible.
_RELEASE_REVISION = "b31d2728952cfaa022bd1df92d82a54c23cf4f8f"


def get_example_dataset(
    out_dir: Union[str, Path] = "example_dataset/",
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
) -> None:
    """Download and extract the bundled example dataset from Hugging Face Hub.

    Retrieves a small prepackaged example dataset used for SRGAN demonstrations
    and tests. The function ensures deterministic extraction by stripping any
    top-level folder prefixes (e.g., ``example_data/``) from the archive so that
    the files always end up directly under the specified output directory.

    Args:
        out_dir (str or Path, optional): Target directory for extraction.
            Defaults to ``"example_dataset/"``.
        cache_dir (str or Path, optional): Hugging Face cache directory.
        revision (str, optional): Hub revision to download. Defaults to the
            immutable package release revision.

    Behaviour:
        1. Creates the output folder if it does not exist.
        2. Downloads ``data/examples/dataset.zip`` from the repository
           ``simon-donike/SR-GAN`` on Hugging Face Hub.
        3. Extracts the archive contents into ``out_dir``, removing any redundant
           root folder structure for cleaner layout.
        4. Leaves Hugging Face's cache-managed archive in place.

    Returns:
        None

    Example:
        >>> get_example_dataset()
        📦 Downloading from Hugging Face Hub...
        ✅ Extracted dataset to: /path/to/example_dataset
    """
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Downloading from Hugging Face Hub...")
    zip_path = hf_hub_download(
        repo_id=_REPO_ID,
        filename=_DATASET_FILENAME,
        cache_dir=None if cache_dir is None else str(cache_dir),
        revision=revision if revision is not None else _RELEASE_REVISION,
    )

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()

        # detect common top-level folder (e.g. "example_data/")
        prefix = os.path.commonprefix(members)
        if prefix and prefix.endswith("/"):
            for member in members:
                # strip the prefix
                target = member[len(prefix) :]
                if not target:  # skip folder itself
                    continue
                _safe_extract_member(z, member, target, output_dir)
        else:
            for member in members:
                _safe_extract_member(z, member, member, output_dir)

    print(f"✅ Extracted dataset to: {output_dir}")
