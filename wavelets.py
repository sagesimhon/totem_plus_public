import json

import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

res = 1024
map_file = '/Users/sage/data-for-totems-new-shapes/r1024_knot_selectbrightest/mappings_cam_to_tot.json'
shape = 'knot-1024_no-interp'
# map_file = '/Users/sage/data-for-totems-new-shapes/r2048-fixed/mappings_cam_to_tot.json'
with open(map_file, 'r') as f:
    mappings = json.load(f)

reverse_mappings = {}
for k, v in mappings.items():
    reverse_mappings.setdefault(v, set()).add(k)
reverse_mappings_len = {}
for k, v in reverse_mappings.items():
    for x in v:
        reverse_mappings_len[x] = reverse_mappings_len.setdefault(x, 0) + len(v)
coords = np.zeros((res, res))
for k, v in reverse_mappings_len.items():
    coord_as_list = [int(num) for num in k.strip('()').split(',')]
    x, y = coord_as_list
    coords[y][x] = v

import matplotlib.pyplot as plt

# cmap = plt.cm.hot
# cmap_reversed = cmap.reversed()

plt.imshow(coords, cmap='hot', interpolation='none')#, interpolation='bicubic')
# plt.imshow(coords, cmap='jet', interpolation='bicubic')
plt.colorbar()  # Optional: adds a colorbar to interpret values
plt.savefig(f'{shape}_compression_map.png')
plt.show()

# makes a weird artifact
# cmap = plt.get_cmap('hot')
# cmap.set_under('yellow')
#
# plt.imshow(coords, cmap=cmap, interpolation='bicubic')
# cbar = plt.colorbar()
# plt.show()


# img = Image.open('/Users/sage/PycharmProjects/totem_new_shapes/data/image_to_unwarp/smiley_face.png').convert('L')
# img = Image.open('/Users/sage/PycharmProjects/totem_new_shapes/data/image_to_unwarp/bird.png').convert('L')
# # img = Image.open('/Users/sage/PycharmProjects/totem_new_shapes/data/image_to_unwarp/stripes-vert.png').convert('L')
#
# image = np.array(img)
#
#
# basis = 'haar'
# coeffs = pywt.dwt2(image, basis)
# cA, (cH, cV, cD) = coeffs
#
# # Visualizing the coefficients
# plt.figure(figsize=(12, 3))
# plt.subplot(1, 4, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image', fontsize=10)
# plt.subplot(1, 4, 2)
# plt.imshow(cA, cmap='gray')
# plt.title('Approximation', fontsize=10)
# plt.subplot(1, 4, 3)
# plt.imshow(cH, cmap='gray')
# plt.title('Horizontal Detail', fontsize=10)
# plt.subplot(1, 4, 4)
# plt.imshow(cV, cmap='gray')
# plt.title('Vertical Detail', fontsize=10)
# plt.show()
#
#
# f_transform = np.fft.fft2(image)
# f_shift = np.fft.fftshift(f_transform)
# #f_shift = f_transform
# magnitude_spectrum = 20*np.log(np.abs(f_shift))
#
# # Visualize the result
# plt.figure(figsize=(12, 6))
# plt.subplot(121), plt.imshow(image, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Fourier Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
#
#
# # Select a wavelet
# wavelet = 'haar'  # You can try different wavelets like 'db1', 'sym5', etc.
#
# # Create subplots
# fig, axes = plt.subplots(2, 5, figsize=(20, 8))
# axes = axes.ravel()
#
# # Original image
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Original Image')
# axes[0].axis('off')
#
# # Perform wavelet decomposition at different scales
# for i in range(1, 10):
#     coeffs = pywt.wavedec2(image, wavelet, level=i)
#     cA, (cH, cV, cD) = coeffs[0], coeffs[1]
#     # Reconstruct image from coefficients
#     reconstructed_image = pywt.waverec2(coeffs, wavelet)
#     # Plot
#     axes[i].imshow(cH, cmap='gray')
#     axes[i].set_title(f'Scale {i}')
#     axes[i].axis('off')
#
# plt.tight_layout()
# plt.show()
#
#
#
# # Define scales
# scale = 10
# scales = np.arange(1, scale + 1)  # 20 scales
#
# # Choose a wavelet
# wavelet = 'shan'  # cmor for Complex Morlet wavelet (shan for shannon wavelet)
#
# # Initialize an array to store CWT results
# cwt_array = np.zeros((len(scales), *image.shape))
#
# # Apply CWT to each row
# for i, row in enumerate(image):
#     coefficients, _ = pywt.cwt(row, scales, wavelet)
#     cwt_array[:, i, :] = np.abs(coefficients)
#
# for i, row in enumerate(image):
#     coefficients, _ = pywt.cwt(row, scales, wavelet)
#     cwt_array[:, i, :] = np.abs(coefficients)
# # Visualization of CWT at each scale
#
# fig, axes = plt.subplots(2, int(scale / 2), figsize=(scale, 8))  # Adjust subplot layout as needed
# axes = axes.flatten()
#
# for i in range(len(scales)):
#     ax = axes[i]
#     ax.imshow(cwt_array[i, :, :], extent=[0, image.shape[1], 0, image.shape[0]], cmap='hot', aspect='auto')
#     ax.set_title(f'Scale {scales[i]}')
#     ax.set_xlabel('Position')
#     ax.set_ylabel('Row')
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#
# plt.tight_layout()
# plt.show()