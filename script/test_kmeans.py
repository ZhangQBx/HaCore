import numpy as np
import random

def kmeans(distances, k, max_iters=100):
    # 初始化聚类中心
    centroids = random.sample(range(len(distances[0])), k)
    for _ in range(max_iters):
        # 分配每个点到最近的聚类中心
        labels = []
        for id in range(len(distances)):
            min_dis = 1000000
            min_centroid = None
            for centroid in centroids:
                if distances[id][centroid] < min_dis:
                    min_dis = distances[id][centroid]
                    min_centroid = centroid
            labels.append(min_centroid)
        distinct_labels = set(labels)
        labels = np.array(labels)
        # 更新聚类中心
        new_centroids = []
        for label in distinct_labels:
            sample_index = np.where(labels == label)[0]
            min_dis = 1000000
            min_centroid = None
            for id1 in sample_index:
                dis_count = 0
                for id2 in sample_index:
                    if id2 != id1:
                        dis_count += distances[id1][id2]
                if dis_count < min_dis:
                    min_dis = dis_count
                    min_centroid = id1
            new_centroids.append(min_centroid)
        # 检查聚类中心是否收敛
        # if np.allclose(new_centroids, centroids):
        #     break

        centroids = new_centroids
        print(_)
    return centroids, labels


def generate_symmetric_random_matrix(n):
    # 生成一个随机数矩阵
    random_matrix = np.random.rand(n, n)

    # 将随机数矩阵转换为对称矩阵
    symmetric_matrix = (random_matrix + random_matrix.T) / 2

    # 将主对角线上的元素设置为0
    np.fill_diagonal(symmetric_matrix, 0)
    return symmetric_matrix

symmetric_matrix = generate_symmetric_random_matrix(50)


# 调用自定义K均值算法
centroids, labels = custom_kmeans(symmetric_matrix, k=5)

# 输出聚类结果
print("Cluster centers:")
print(centroids)
print("\nLabels:")
print(labels)
