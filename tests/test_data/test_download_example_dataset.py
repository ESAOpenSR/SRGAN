import zipfile

from opensr_srgan.data.example_data import download_example_dataset
from opensr_srgan import _factory


def test_example_dataset_uses_the_preset_release_revision():
    assert (
        download_example_dataset._RELEASE_REVISION
        == _factory._PRESETS["SEN2-SR"].revision
    )


def test_get_example_dataset_uses_structured_pinned_hub_asset(monkeypatch, tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("example_data/", b"")
        zip_file.writestr("example_data/hr_000.npz", b"example-chip")

    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return archive

    monkeypatch.setattr(
        download_example_dataset, "hf_hub_download", fake_hf_hub_download
    )
    output_dir = tmp_path / "extracted"

    download_example_dataset.get_example_dataset(
        output_dir, cache_dir=tmp_path / "cache"
    )

    assert calls == [
        {
            "repo_id": "simon-donike/SR-GAN",
            "filename": "data/examples/dataset.zip",
            "cache_dir": str(tmp_path / "cache"),
            "revision": download_example_dataset._RELEASE_REVISION,
        }
    ]
    assert (output_dir / "hr_000.npz").read_bytes() == b"example-chip"


def test_get_example_dataset_allows_explicit_revision(monkeypatch, tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("hr_000.npz", b"example-chip")

    call = {}

    def fake_hf_hub_download(**kwargs):
        call.update(kwargs)
        return archive

    monkeypatch.setattr(
        download_example_dataset, "hf_hub_download", fake_hf_hub_download
    )

    download_example_dataset.get_example_dataset(
        tmp_path / "extracted", revision="test"
    )

    assert call["revision"] == "test"
    assert call["cache_dir"] is None
