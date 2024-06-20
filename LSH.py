from datasketch import MinHash, MinHashLSH
import pandas as pd
import numpy as np
import random

df = pd.read_csv('movies_with_genres.csv')
# Dữ liệu mẫu
movies = {}

for index, row in df.iterrows():
    title = row['title']
    tags = str(row['tag']).split(', ') 
    tags = [tag.strip() for tag in tags]
    movies[title] = tags
    

def minhash(shingles, num_hashes):
    num_shingles = len(shingles)

    # Khởi tạo một mảng để lưu trữ các giá trị băm nhỏ nhất cho mỗi hoán vị
    min_hashes = [float('inf')] * num_hashes

    # Tạo các hoán vị ngẫu nhiên của tập hợp shingles
    for _ in range(num_hashes):
        permutation = list(range(num_shingles))
        random.shuffle(permutation)

        # Tính toán giá trị băm của mỗi shingle trong hoán vị
        for i, shingle_index in enumerate(permutation):
            shingle = shingles[shingle_index]
            hash_value = hash(shingle)  # Sử dụng hàm băm tích hợp của Python

            # Cập nhật giá trị băm nhỏ nhất nếu cần
            if hash_value < min_hashes[i]:
                min_hashes[i] = hash_value

    return min_hashes
    
def simhash(shingles, num_bits):
    v = np.zeros(num_bits)
    for shingle in shingles:
        hash_value = hash(shingle)
        for i in range(num_bits):
            bit = (hash_value >> i) & 1
            v[i] += (2 * bit - 1)
    return "".join(['1' if x > 0 else '0' for x in v])

def lsh_tables(signatures, num_bands, num_rows, tables=None):
    if tables is None:
        tables = {}

    for i, signature in enumerate(signatures):
        band_index = i // num_rows  # Tính chỉ số dải
        if band_index not in tables:
            tables[band_index] = {}
        table = tables[band_index]
        if signature not in table:
            table[signature] = []
        table[signature].append(i)  # Lưu trữ chỉ số của signature trong bảng băm

    return tables

def lsh_query(query_signatures, tables, num_bands):
    candidates = set()
    for i, signature in enumerate(query_signatures):
        band_index = i // (len(query_signatures) // num_bands)
        if band_index in tables:
            table = tables[band_index]
            if signature in table:
                candidates.update(table[signature])
    return candidates


num_hashes = 512
num_bits = 256

minhash_tables = {}
simhash_tables = {}
lsh = MinHashLSH(threshold=0.5, num_perm=num_hashes)

for title, tags in movies.items():
    minhash_signatures = minhash(tags, num_hashes)
    simhash_signature = simhash(tags, num_bits)

    # Thêm vào bảng băm MinHash (bạn cần tự định nghĩa hàm lsh_tables)
    minhash_tables = lsh_tables(minhash_signatures, 16, 8, minhash_tables) 

    # Thêm vào bảng băm SimHash (bạn cần tự định nghĩa hàm lsh_tables)
    simhash_tables = lsh_tables(simhash_signature, 16, 8, simhash_tables) 

query = ['animation', 'comedy']
minhash_candidates = lsh_query(minhash(query, num_hashes), minhash_tables, 16)
simhash_candidates = lsh_query(simhash(query, num_bits), simhash_tables, 16)

print("Ứng viên MinHash:", len(minhash_candidates))
print("Ứng viên SimHash:", len(simhash_candidates))
