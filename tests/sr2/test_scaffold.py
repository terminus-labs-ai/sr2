def test_version_exists():
    import sr2

    assert hasattr(sr2, "__version__")
    assert sr2.__version__ == "0.1.0"
