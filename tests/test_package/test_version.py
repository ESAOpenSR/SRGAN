from opensr_srgan import __version__


def test_version_is_a_string():
    assert isinstance(__version__, str)
    assert __version__
