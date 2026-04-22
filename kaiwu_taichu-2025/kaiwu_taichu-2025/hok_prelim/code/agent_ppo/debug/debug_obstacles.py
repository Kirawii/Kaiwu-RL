import cv2
import numpy as np

img = cv2.imread("/data/projects/back_to_the_realm_v2/agent_ppo/debug/map/map_raw1.png", cv2.IMREAD_GRAYSCALE)
print(img)
# big_img = np.zeros((51, 51), np.uint8)
# for i in range(51):
#     for j in range(51):
#         ii = (i - 25) // 5 + 5
#         jj = (j - 25) // 5 + 5
#         if ii < 0 or ii >= 10 or jj < 0 or jj >= 10:
#             continue
#         big_img[i, j] = img[ii, jj]
# print(big_img)
# big_img = cv2.resize(img, (51, 51), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite("/data/projects/back_to_the_realm_v2/agent_ppo/debug/big_map.png", big_img)
img[10, 10] = 123
print(img[img == 123])

print(np.uint8(-255))
