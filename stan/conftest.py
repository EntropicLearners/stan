import pathlib
import pytest


@pytest.fixture(scope='session')
def get_file():
    test_dir = pathlib.Path(__file__).parent.resolve()
    def _get_file(fn):
        return test_dir / f'tests/data/{fn}'
    return _get_file


@pytest.fixture(scope="function", autouse=True)
def output_cleanup(request):
    print('\n-----------------')
    print('function    : %s' % request.function.__name__)
    print('-----------------')