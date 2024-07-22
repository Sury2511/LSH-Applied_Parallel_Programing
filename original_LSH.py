from datasketch import MinHash, MinHashLSH
import pandas as pd
import numpy as np
import random
import time
from concurrent.futures import ThreadPoolExecutor
from numba import cuda
import os

start = time.time()
df = pd.read_csv('movies_with_genres.csv')
movies = {}
for index, row in df.iterrows():
    title = row['originalTitle']
    tags = str(row['genres']).split(', ')
    tags = [tag.strip() for tag in tags]
    movies[title] = tags


num_hashes = 128  

lsh = MinHashLSH(threshold=0.5, num_perm=num_hashes)

for title, tags in movies.items():
    m = MinHash(num_perm=num_hashes)
    for tag in tags:
        m.update(tag.encode('utf8'))
    lsh.insert(title, m)  


end = time.time()
print(f"Thời gian tạo bảng băm LSH: {end - start:.2f} giây")

while True:
    query_str = input("Nhập truy vấn (các thể loại phim, cách nhau bằng dấu phẩy): ")
    
    print(end - start)
    if not query_str:
        break  # Thoát nếu người dùng nhập chuỗi rỗng
    query = [tag.strip() for tag in query_str.split(',')]

    # Tạo đối tượng MinHash cho truy vấn
    query_minhash = MinHash(num_perm=num_hashes)
    for tag in query:
        query_minhash.update(tag.encode('utf8'))

    result = lsh.query(query_minhash)

    # In kết quả
    if result:
        print("Các bộ phim tương tự:")
        for title in result:
            print(f"- {title}: {movies[title]}")
    else:
        print("Không tìm thấy bộ phim tương tự.")
    print()
