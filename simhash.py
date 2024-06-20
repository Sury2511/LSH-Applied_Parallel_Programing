import pandas as pd
import numpy as np
import time

# Đọc dữ liệu từ file CSV
df = pd.read_csv('movies_with_genres.csv')
movies = {}
for index, row in df.iterrows():
    title = row['title']
    tags = str(row['tag']).split(', ')
    tags = [tag.strip() for tag in tags]
    movies[title] = tags

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
        band_index = i // num_rows
        if band_index not in tables:
            tables[band_index] = {}
        table = tables[band_index]
        if signature not in table:
            table[signature] = []
        table[signature].append(i)

    return tables

def lsh_query(query_signature, tables, num_bands):
    candidates = set()
    band_index = hash(query_signature) % num_bands  
    if band_index in tables:
        table = tables[band_index]
        if query_signature in table:
            candidates.update(table[query_signature])
    return candidates

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

num_bits = 256
num_bands = 8
num_rows = 16

simhash_tables = {}
start_time = time.time()

for title, tags in movies.items():
    simhash_signature = simhash(tags, num_bits)
    simhash_tables = lsh_tables(simhash_signature, num_bands, num_rows, simhash_tables)

end_time = time.time()
print(f"Thời gian tạo bảng băm LSH: {end_time - start_time:.2f} giây")

# Thử nghiệm truy vấn
while True:
    query_str = input("Nhập truy vấn (các thể loại phim, cách nhau bằng dấu phẩy): ")
    if not query_str:
        break  # Thoát nếu người dùng nhập chuỗi rỗng
    query = [tag.strip() for tag in query_str.split(',')]

    # Truy vấn SimHash
    query_simhash = simhash(query, num_bits)
    candidate_indices = lsh_query(query_simhash, simhash_tables, num_bands)
    candidates = [list(movies.keys())[i] for i in candidate_indices]

    # Đánh giá độ tương đồng và lọc các ứng viên
    similar_movies = []
    for candidate in candidates:
        candidate_vec = np.array([int(bit) for bit in simhash(movies[candidate], num_bits)])
        query_vec = np.array([int(bit) for bit in query_simhash])
        similarity = cosine_similarity(query_vec, candidate_vec)
        if similarity >= 0.8:  # Ngưỡng bạn có thể tùy chỉnh
            similar_movies.append((candidate, similarity))

    # Sắp xếp các bộ phim tương tự theo độ tương đồng giảm dần
    similar_movies.sort(key=lambda x: x[1], reverse=True)

    # In kết quả
    if similar_movies:
        print("Các bộ phim tương tự:")
        for title, similarity in similar_movies:
            print(f"- {title}: {', '.join(movies[title])} (Độ tương đồng: {similarity:.2f})")
    else:
        print("Không tìm thấy bộ phim tương tự.")
    print()
