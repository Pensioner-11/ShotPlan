from diffsynth.core import UnifiedDataset


class FlexibleDataset(UnifiedDataset):
    """UnifiedDataset variant whose metadata loading and per-field operators
    are supplied by a data profile (see data_profiles.py)."""

    def __init__(self, data_profile, base_path, metadata_path, *args, **kwargs):
        self.profile = data_profile
        # Pass metadata_path=None so the parent class does not try to load it.
        super().__init__(metadata_path=None, *args, **kwargs)
        self.load_from_cache = False

        self.data = self.profile.load_and_transform(metadata_path)
        self.special_operator_map = self.profile.get_operator_map()
        self.data_file_keys = self.profile.get_data_keys()

        print(f"[Dataset] Initialized with profile: {self.profile.__class__.__name__}")
        print(f"[Dataset] Total samples: {len(self.data)}")

    def load_metadata(self, metadata_path):
        pass

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data
