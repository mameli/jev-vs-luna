"""Keep paid network checks opt-in, even when .env contains a key."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="Enable paid API integration tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        for item in items:
            if "live" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --live to enable API calls"))
