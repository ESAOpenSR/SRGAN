from pathlib import Path
import tomllib

import yaml

from opensr_srgan import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_version_is_a_string():
    assert isinstance(__version__, str)
    assert __version__


def test_project_and_citation_metadata_are_synchronized():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    citation = yaml.safe_load((PROJECT_ROOT / "CITATION.cff").read_text())

    assert citation["version"] == project["version"]
    assert citation["repository-code"] == project["urls"]["Homepage"]
