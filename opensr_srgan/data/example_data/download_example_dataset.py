import os
from pathlib import Path
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


def get_example_dataset(out_dir: str = "example_dataset/"):
    """Download and extract the bundled example dataset from Hugging Face Hub.

    Retrieves a small prepackaged example dataset used for SRGAN demonstrations
    and tests. The function ensures deterministic extraction by stripping any
    top-level folder prefixes (e.g., ``example_data/``) from the archive so that
    the files always end up directly under the specified output directory.

    Args:
        out_dir (str, optional): Target directory for extraction.
            Defaults to ``"example_dataset/"``.

    Behaviour:
        1. Creates the output folder if it does not exist.
        2. Downloads ``example_dataset.zip`` from the repository
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

    repo_id = "simon-donike/SR-GAN"
    filename = "example_dataset.zip"

    print("📦 Downloading from Hugging Face Hub...")
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename)

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
