import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules import extensions


class ExtensionRepoMetadataCacheTest(unittest.TestCase):
    def setUp(self):
        extensions.clear_repo_metadata_cache()

    def tearDown(self):
        extensions.clear_repo_metadata_cache()

    def test_read_info_from_repo_reuses_process_cache_for_fresh_extension_objects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir) / "extension"
            git_dir = extension_dir / ".git"
            git_dir.mkdir(parents=True)

            calls = []
            repo_data = {
                "remote": "https://example.test/repo.git",
                "commit_date": 123,
                "branch": "main",
                "commit_hash": "abcdef123456",
                "version": "abcdef12",
            }

            def cached_data_for_file(_subsection, _title, _filename, func):
                calls.append(_filename)
                value = func()
                return value if value is not None else repo_data

            first = extensions.Extension("sample", str(extension_dir), metadata=extensions.ExtensionMetadata(str(extension_dir), "sample"))
            second = extensions.Extension("sample", str(extension_dir), metadata=extensions.ExtensionMetadata(str(extension_dir), "sample"))

            with mock.patch.object(extensions.cache, "cached_data_for_file", side_effect=cached_data_for_file):
                with mock.patch.object(first, "do_read_info_from_repo", side_effect=lambda: [setattr(first, field, value) for field, value in repo_data.items()] and setattr(first, "have_info_from_repo", True)):
                    first.read_info_from_repo()

                second.read_info_from_repo()

            self.assertEqual(1, len(calls))
            self.assertEqual(repo_data["remote"], second.remote)
            self.assertEqual(repo_data["branch"], second.branch)
            self.assertEqual(repo_data["commit_hash"], second.commit_hash)
            self.assertTrue(second.have_info_from_repo)


if __name__ == "__main__":
    unittest.main()
