def test_import():
    import magie



def test_all_submodules_importable():
    # tests/test_imports.py
    import importlib
    import pkgutil

    import magie

    failures = []

    for module_info in pkgutil.walk_packages(
        magie.__path__,
        prefix=magie.__name__ + ".",
    ):
        module_name = module_info.name
        if ''

        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append((module_name, repr(exc)))

    assert not failures, "Failed imports:\n" + "\n".join(
        f"{name}: {error}" for name, error in failures
    )