from modules import model_cache_budget


class FakeTensor:
    def __init__(self, numel, element_size):
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class FakeCheckpointInfo:
    def __init__(self, title):
        self.title = title


class FakeModel:
    def __init__(self, title, tensors):
        self.sd_checkpoint_info = FakeCheckpointInfo(title)
        self._parameters = list(tensors)
        self._buffers = []

    def parameters(self):
        return iter(self._parameters)

    def buffers(self):
        return iter(self._buffers)


def make_model(title, size):
    return FakeModel(title, [FakeTensor(size, 1)])


def test_estimate_model_bytes_counts_parameters_and_buffers_once():
    shared_tensor = FakeTensor(10, 2)
    model = FakeModel("model", [shared_tensor, shared_tensor, FakeTensor(4, 4)])
    model._buffers = [FakeTensor(2, 8)]

    assert model_cache_budget.estimate_model_bytes(model) == 52


def test_select_models_to_evict_uses_lru_order_for_count_limit():
    newest = make_model("newest", 10)
    middle = make_model("middle", 10)
    oldest = make_model("oldest", 10)

    evicted = model_cache_budget.select_models_to_evict([newest, middle, oldest], max_count=2)

    assert evicted == [oldest]


def test_select_models_to_evict_respects_byte_budget_and_protected_model():
    current = make_model("current", 70)
    inactive = make_model("inactive", 50)
    old = make_model("old", 40)

    evicted = model_cache_budget.select_models_to_evict(
        [current, inactive, old],
        protected_models=[current],
        max_bytes=100,
    )

    assert evicted == [old, inactive]


def test_can_add_model_to_cache_accounts_for_existing_model_size():
    model = make_model("current", 80)

    assert model_cache_budget.can_add_model_to_cache([model], max_count=2, max_bytes=200)
    assert not model_cache_budget.can_add_model_to_cache([model], max_count=2, max_bytes=120)
    assert not model_cache_budget.can_add_model_to_cache([model], max_count=1, max_bytes=0)
