from datasketch import MinHash, MinHashLSH
import pandas as pd
import numpy as np
import random

# Đọc dữ liệu từ file CSV
df = pd.read_csv('movies_with_genres.csv')
movies = {}
for index, row in df.iterrows():
    title = row['title']
    tags = str(row['tag']).split(', ')
    tags = [tag.strip() for tag in tags]
    movies[title] = tags

# Hàm MinHash
def minhash(shingles, num_hashes):
    num_shingles = len(shingles)
    min_hashes = [float('inf')] * num_hashes
    for _ in range(num_hashes):
        permutation = list(range(num_shingles))
        random.shuffle(permutation)
        for i, shingle_index in enumerate(permutation):
            shingle = shingles[shingle_index]
            hash_value = hash(shingle)
            if hash_value < min_hashes[i]:
                min_hashes[i] = hash_value
    return min_hashes

# Hàm SimHash
def simhash(shingles, num_bits):
    v = np.zeros(num_bits)
    for shingle in shingles:
        hash_value = hash(shingle)
        for i in range(num_bits):
            bit = (hash_value >> i) & 1
            v[i] += (2 * bit - 1)
    return "".join(['1' if x > 0 else '0' for x in v])

# Hàm tạo bảng băm LSH
def lsh_tables(signatures, num_bands, num_rows, tables=None):
    if tables is None:
        tables = {}

    for i, signature in enumerate(signatures):
        band_index = i // num_rows
        if band_index not in tables:
            tables[band_index] = {}
        table = tables[band_index]
        if signature not in table:
            table[signature] = []
        table[signature].append(i)

    return tables

# Hàm truy vấn LSH
def lsh_query(query_signatures, tables, num_bands):
    candidates = set()
    for i, signature in enumerate(query_signatures):
        band_index = i // (len(query_signatures) // num_bands)
        if band_index in tables:
            table = tables[band_index]
            if signature in table:
                candidates.update(table[signature])
    return candidates

# Hàm tính Jaccard similarity (cho MinHash)
def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

# Hàm tính cosine similarity (cho SimHash)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Tham số
num_hashes = 512
num_bits = 256

# Tạo bảng băm LSH và thêm các MinHash vào
minhash_tables = {}
simhash_tables = {}
lsh = MinHashLSH(threshold=0.5, num_perm=num_hashes)

for title, tags in movies.items():
    # MinHash
    m = MinHash(num_perm=num_hashes)
    for tag in tags:
        m.update(tag.encode('utf8'))
    lsh.insert(title, m)  
    minhash_signatures = minhash(tags, num_hashes)
    minhash_tables = lsh_tables(minhash_signatures, 16, 8, minhash_tables) 

    # SimHash
    simhash_signature = simhash(tags, num_bits)
    simhash_tables = lsh_tables(simhash_signature, 16, 8, simhash_tables) 


query = ['animation', 'comedy']


query_minhash = MinHash(num_perm=num_hashes)
for tag in query:
    query_minhash.update(tag.encode('utf8'))
minhash_candidates = lsh.query(query_minhash)

simhash_candidates = lsh_query(simhash(query, num_bits), simhash_tables, 16)
simhash_candidates = [list(movies.keys())[i] for i in simhash_candidates]

minhash_precision = 0
if len(minhash_candidates) > 0: 
    for candidate in minhash_candidates:
        similarity = jaccard_similarity(set(query), set(movies[candidate]))
        if similarity >= 0.5:
            minhash_precision += 1
    minhash_precision /= len(minhash_candidates)
else:
    minhash_precision = 0  
    
simhash_precision = 0
for candidate in simhash_candidates:
    # Lấy đúng chuỗi bit SimHash từ simhash_tables
    for band_index, band_table in simhash_tables.items():
        if simhash(movies[candidate], num_bits) in band_table:
            candidate_vec = np.array([int(bit) for bit in simhash(movies[candidate], num_bits)])
            break  
    else:
        continue 

    query_vec = np.array([int(bit) for bit in simhash(query, num_bits)])
    similarity = cosine_similarity(query_vec, candidate_vec)
    if similarity >= 0.8:
        simhash_precision += 1

simhash_precision /= len(simhash_candidates)

print("Precision MinHash:", minhash_precision)
print("Precision SimHash:", simhash_precision)
