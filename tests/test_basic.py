import shppy
import pytest

class TestBasic:
    def test_add_floats(self):
        assert shppy.add_floats(1.0, 2.0) == 3.0

if __name__ == "__main__":
    pytest.main()
