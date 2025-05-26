import pytest

def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")

@pytest.fixture(scope='session')
def device(request):
    name_value = request.config.option.device
    if name_value is None:
        pytest.skip()
    return name_value
