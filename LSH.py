import pandas as pd
import numpy as np
import time
from numba import cuda, jit
import LSH_class

# Đọc file CSV
df = pd.read_csv('movies_with_genres.csv')
print(df.shape)
movies = {}
for index, row in df.iterrows():
    title = row['originalTitle']
    tags = str(row['genres']).split(', ')
    tags = [tag.strip() for tag in tags]
    movies[title] = tags

@jit
def custom_hash(x, seed):
    return (x * 0x9e3779b9 + seed) & 0xFFFFFFFF

@cuda.jit
def cuda_update(tags, minhashes, num_hashes):
    pos = cuda.grid(1)
    if pos < tags.shape[0]:
        tag = tags[pos]
        for i in range(num_hashes):
            tag_hash = custom_hash(tag, i)
            cuda.atomic.min(minhashes, (pos, i), tag_hash)

def update_minhash_gpu(tags, num_hashes):
    tags_gpu = cuda.to_device(np.array(tags, dtype=np.int32))
    num_tags = tags_gpu.size
    minhashes_gpu = cuda.to_device(np.full((num_tags, num_hashes), np.inf, dtype=np.float32))

    threads_per_block = 256
    blocks_per_grid = (num_tags + (threads_per_block - 1)) // threads_per_block

    cuda_update[blocks_per_grid, threads_per_block](tags_gpu, minhashes_gpu, num_hashes)

    return minhashes_gpu.copy_to_host()

unique_tags = set(tag for tags in movies.values() for tag in tags)
tag_to_int = {tag: i for i, tag in enumerate(unique_tags)}

num_hashes = 128
start = time.time()
lsh = LSH_class.LSH(num_hashes=num_hashes)

int_tags_dict = {title: np.array([tag_to_int[tag] for tag in tags], dtype=np.int32) for title, tags in movies.items()}
all_tags = np.concatenate(list(int_tags_dict.values()))
tags_offset = np.cumsum([0] + [len(tags) for tags in int_tags_dict.values()])

all_tags_gpu = cuda.to_device(all_tags)
minhashes_gpu = cuda.to_device(np.full((all_tags.size, num_hashes), np.inf, dtype=np.float32))

threads_per_block = 256
blocks_per_grid = (all_tags.size + (threads_per_block - 1)) // threads_per_block

cuda_update[blocks_per_grid, threads_per_block](all_tags_gpu, minhashes_gpu, num_hashes)

# Lấy minhashes từ GPU
all_minhashes = minhashes_gpu.copy_to_host()

# Tạo MinHash signatures và chèn vào LSH
for i, (title, tags) in enumerate(movies.items()):
    start_idx = tags_offset[i]
    end_idx = tags_offset[i + 1]
    minhash_signature = np.min(all_minhashes[start_idx:end_idx], axis=0)
    lsh.insert(title, minhash_signature)

end = time.time()
print(f"Thời gian tạo bảng băm LSH: {end - start:.2f} giây")

# Truy vấn
while True:
    query_str = input("Nhập truy vấn (các thể loại phim, cách nhau bằng dấu phẩy): ")
    if not query_str:
        break
    query = [tag.strip() for tag in query_str.split(',')]

    int_query = [tag_to_int[tag] for tag in query if tag in tag_to_int]
    if not int_query:
        print("Không tìm thấy thể loại phim nào trong truy vấn.")
        continue

    minhashes = update_minhash_gpu(int_query, num_hashes)
    query_minhash = np.min(minhashes, axis=0)

    result = lsh.query(query_minhash)

    if result:
        print("Các bộ phim tương tự:")
        for title in result:
            print(f"- {title}: {movies[title]}")
    else:
        print("Không tìm thấy bộ phim tương tự.")
    print()
