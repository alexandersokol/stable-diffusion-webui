import tempfile
import sys
import types
import json
from pathlib import Path
from types import SimpleNamespace


images_stub = types.ModuleType("modules.images")
images_stub.read_info_from_image = lambda image: (None, {})
images_stub.save_image_with_geninfo = lambda image, geninfo, filename: None
sys.modules.setdefault("modules.images", images_stub)

infotext_utils_stub = types.ModuleType("modules.infotext_utils")
infotext_utils_stub.image_from_url_text = lambda text: None
sys.modules.setdefault("modules.infotext_utils", infotext_utils_stub)

user_metadata_stub = types.ModuleType("modules.ui_extra_networks_user_metadata")


class StubUserMetadataEditor:
    def __init__(self, *args, **kwargs):
        pass


user_metadata_stub.UserMetadataEditor = StubUserMetadataEditor
sys.modules.setdefault("modules.ui_extra_networks_user_metadata", user_metadata_stub)

from modules import shared, ui_extra_networks


class FakeExtraNetworksPage(ui_extra_networks.ExtraNetworksPage):
    def __init__(self, root, count):
        super().__init__("Fake")
        self.root = root
        self.count = count
        self.metadata_reads = []

    def list_items(self):
        for index in range(self.count):
            filename = str(self.root / f"item-{index}.pt")
            yield {
                "name": f"item-{index}",
                "filename": filename,
                "shorthash": f"hash-{index}",
                "preview": None,
                "description": f"description for item-{index}",
                "search_terms": [f"item-{index}", f"group-{index % 2}"],
                "prompt": ui_extra_networks.quote_js(f"item-{index}"),
                "local_preview": str(self.root / f"item-{index}.preview.png"),
                "sort_keys": {"default": index, "name": f"item-{index}", "date_created": index, "date_modified": index},
            }

    def read_user_metadata(self, item, use_cache=True):
        self.metadata_reads.append(item["name"])
        item["user_metadata"] = {}

    def allowed_directories_for_previews(self):
        return [str(self.root)]


def patch_opts(page_size=2):
    original_opts = shared.opts
    shared.opts = SimpleNamespace(
        extra_networks_card_height=0,
        extra_networks_card_width=0,
        extra_networks_card_text_scale=1.0,
        extra_networks_card_show_desc=True,
        extra_networks_card_description_is_html=False,
        extra_networks_hidden_models="Always",
        extra_networks_card_page_size=page_size,
        extra_networks_tree_view_default_enabled=False,
        extra_networks_card_order="Ascending",
        extra_networks_card_order_field="Path",
        extra_networks_tree_view_default_width=180,
        extra_networks_tree_view_style="Dirs",
        extra_networks_dir_button_function=False,
        extra_networks_show_hidden_directories=False,
        samples_format="png",
    )
    return original_opts


def test_create_html_renders_only_first_card_batch_and_defers_metadata():
    with tempfile.TemporaryDirectory() as temp_dir:
        original_opts = patch_opts(page_size=2)
        try:
            page = FakeExtraNetworksPage(Path(temp_dir), 5)

            html = page.create_html("txt2img")

            assert 'data-name="item-0"' in html
            assert 'data-name="item-1"' in html
            assert 'data-name="item-2"' not in html
            assert "extra-networks-load-more" in html
            assert page.metadata_reads == ["item-0", "item-1"]
        finally:
            shared.opts = original_opts


def test_create_card_batch_html_renders_next_batch_and_updates_offset():
    with tempfile.TemporaryDirectory() as temp_dir:
        original_opts = patch_opts(page_size=2)
        try:
            page = FakeExtraNetworksPage(Path(temp_dir), 5)
            page.create_html("txt2img")

            html, next_offset, total = page.create_card_batch_html("txt2img", offset=2, limit=2, include_load_more=True)

            assert 'data-name="item-2"' in html
            assert 'data-name="item-3"' in html
            assert 'data-name="item-4"' not in html
            assert "Load more (1)" in html
            assert next_offset == 4
            assert total == 5
            assert page.metadata_reads == ["item-0", "item-1", "item-2", "item-3"]
        finally:
            shared.opts = original_opts


def test_create_card_batch_html_zero_page_size_renders_all_cards():
    with tempfile.TemporaryDirectory() as temp_dir:
        original_opts = patch_opts(page_size=0)
        try:
            page = FakeExtraNetworksPage(Path(temp_dir), 3)

            html = page.create_html("txt2img")

            assert 'data-name="item-0"' in html
            assert 'data-name="item-1"' in html
            assert 'data-name="item-2"' in html
            assert "extra-networks-load-more" not in html
            assert page.metadata_reads == ["item-0", "item-1", "item-2"]
        finally:
            shared.opts = original_opts


def test_create_card_batch_html_filters_search_before_paging():
    with tempfile.TemporaryDirectory() as temp_dir:
        original_opts = patch_opts(page_size=2)
        try:
            page = FakeExtraNetworksPage(Path(temp_dir), 6)
            page.create_html("txt2img")

            html, next_offset, total = page.create_card_batch_html(
                "txt2img",
                offset=0,
                limit=2,
                include_load_more=True,
                search="group-1",
            )

            assert total == 3
            assert next_offset == 2
            assert 'data-name="item-1"' in html
            assert 'data-name="item-3"' in html
            assert 'data-name="item-5"' not in html
            assert "Load more (1)" in html
        finally:
            shared.opts = original_opts


def test_get_page_cards_sorts_server_side_before_paging():
    with tempfile.TemporaryDirectory() as temp_dir:
        original_opts = patch_opts(page_size=2)
        original_pages = list(ui_extra_networks.extra_pages)
        try:
            page = FakeExtraNetworksPage(Path(temp_dir), 4)
            ui_extra_networks.extra_pages.clear()
            ui_extra_networks.extra_pages.append(page)
            page.create_html("txt2img")

            response = ui_extra_networks.get_page_cards(
                page="fake",
                tabname="txt2img",
                offset=0,
                limit=2,
                sort="name",
                sort_dir="Descending",
            )
            payload = json.loads(response.body)

            assert payload["offset"] == 0
            assert payload["next_offset"] == 2
            assert payload["total"] == 4
            assert payload["complete"] is False
            assert payload["html"].index('data-name="item-3"') < payload["html"].index('data-name="item-2"')
            assert 'data-name="item-1"' not in payload["html"]
            assert 'data-name="item-0"' not in payload["html"]
        finally:
            ui_extra_networks.extra_pages.clear()
            ui_extra_networks.extra_pages.extend(original_pages)
            shared.opts = original_opts


def test_get_page_cards_returns_json_for_next_batch():
    with tempfile.TemporaryDirectory() as temp_dir:
        original_opts = patch_opts(page_size=2)
        original_pages = list(ui_extra_networks.extra_pages)
        try:
            page = FakeExtraNetworksPage(Path(temp_dir), 4)
            ui_extra_networks.extra_pages.clear()
            ui_extra_networks.extra_pages.append(page)
            page.create_html("txt2img")

            response = ui_extra_networks.get_page_cards(page="fake", tabname="txt2img", offset=2, limit=2)
            payload = json.loads(response.body)

            assert payload["offset"] == 2
            assert payload["next_offset"] == 4
            assert payload["total"] == 4
            assert payload["complete"] is True
            assert 'data-name="item-2"' in payload["html"]
            assert 'data-name="item-3"' in payload["html"]
        finally:
            ui_extra_networks.extra_pages.clear()
            ui_extra_networks.extra_pages.extend(original_pages)
            shared.opts = original_opts
