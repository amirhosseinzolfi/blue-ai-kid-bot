import unittest

class TestLangMem(unittest.TestCase):
    def test_import_langmem(self):
        try:
            from langmem import LangMem
            self.assertIsNotNone(LangMem)
        except ImportError:
            self.fail("ImportError: LangMem could not be imported from langmem")

if __name__ == '__main__':
    unittest.main()