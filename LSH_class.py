class LSH:
    def __init__(self, num_hashes, threshold=0.5):
        self.num_hashes = num_hashes
        self.threshold = threshold
        self.hash_tables = [dict() for _ in range(num_hashes)]
    
    def _get_bands(self, minhash):
        bands = []
        rows_per_band = self.num_hashes // len(self.hash_tables)
        for i in range(len(self.hash_tables)):
            start = i * rows_per_band
            end = (i + 1) * rows_per_band
            band = tuple(minhash[start:end])
            bands.append(band)
        return bands
    
    def insert(self, key, minhash):
        bands = self._get_bands(minhash)
        for i, band in enumerate(bands):
            if band not in self.hash_tables[i]:
                self.hash_tables[i][band] = []
            self.hash_tables[i][band].append(key)
    
    def query(self, minhash):
        bands = self._get_bands(minhash)
        candidates = set()
        for i, band in enumerate(bands):
            if band in self.hash_tables[i]:
                candidates.update(self.hash_tables[i][band])
        return list(candidates)
    def show_signatures(self):
        signatures = {}
        for i, table in enumerate(self.hash_tables):
            for band, keys in table.items():
                for key in keys:
                    if key not in signatures:
                        signatures[key] = []
                    signatures[key].append(band)
        for key, bands in signatures.items():
            print(f"Key: {key}, Bands: {bands}")