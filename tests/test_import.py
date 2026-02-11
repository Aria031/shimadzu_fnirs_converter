def test_import():
    import shimadzu_fnirs_converter
    assert hasattr(shimadzu_fnirs_converter, "__version__")


def test_api_exists():
    from shimadzu_fnirs_converter import convert, ConvertConfig
    assert callable(convert)
    cfg = ConvertConfig()
    assert cfg.length_unit in ("mm", "cm", "m")


def test_command_interface_entry():
    from shimadzu_fnirs_converter.command_interface import main
    assert callable(main)
