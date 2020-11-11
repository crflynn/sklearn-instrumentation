import pytest


@pytest.fixture
def fixture_function():
    def function(*args, **kwargs):
        print("executing")

    return function
