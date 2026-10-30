
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================
# 1. MEMBACA CITRA
# ==================================

img = cv2.imread("cat.jpg")

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

tinggi, lebar = img.shape[:2]

# ==================================
# 2. TRANSLATION
# ==================================

M = np.float32([
    [1, 0, 100],
    [0, 1, 50]
])

translation = cv2.warpAffine(
    img, M, (lebar, tinggi)
)

# ==================================
# 3. SCALING
# ==================================

scaling_05 = cv2.resize(
    img, None,
    fx=0.5, fy=0.5,
    interpolation=cv2.INTER_LINEAR
)

scaling_2 = cv2.resize(
    img, None,
    fx=2, fy=2,
    interpolation=cv2.INTER_LINEAR
)

# ==================================
# 4. ROTATION
# ==================================

pusat = (lebar // 2, tinggi // 2)

# Rotasi 45 derajat
M45 = cv2.getRotationMatrix2D(pusat, 45, 1.0)
rotation_45 = cv2.warpAffine(img, M45, (lebar, tinggi))

# Rotasi 90 derajat
M90 = cv2.getRotationMatrix2D(pusat, 90, 1.0)
rotation_90 = cv2.warpAffine(img, M90, (lebar, tinggi))

# Rotasi 180 derajat
M180 = cv2.getRotationMatrix2D(pusat, 180, 1.0)
rotation_180 = cv2.warpAffine(img, M180, (lebar, tinggi))

# ==================================
# 5. FLIP
# ==================================

horizontal_flip = cv2.flip(img, 1)
vertical_flip = cv2.flip(img, 0)

# ==================================
# 6. MENAMPILKAN SEMUA HASIL
# ==================================

gambar = [
    img,
    translation,
    scaling_05,
    scaling_2,
    rotation_45,
    rotation_90,
    rotation_180,
    horizontal_flip,
    vertical_flip
]

judul = [
    "Citra Asli",
    "Translation",
    "Scaling 0.5",
    "Scaling 2",
    "Rotation 45 Derajat",
    "Rotation 90 Derajat",
    "Rotation 180 Derajat",
    "Horizontal Flip",
    "Vertical Flip"
]

plt.figure(figsize=(15, 12))

for i in range(len(gambar)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(gambar[i])
    plt.title(judul[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# ==================================
# 7. INFORMASI UKURAN CITRA
# ==================================

print("Ukuran Citra Asli:", img.shape)

print("Ukuran Translation:", translation.shape)
print("Ukuran Scaling 0.5:", scaling_05.shape)
print("Ukuran Scaling 2:", scaling_2.shape)

print("Ukuran Rotation 45:", rotation_45.shape)
print("Ukuran Rotation 90:", rotation_90.shape)
print("Ukuran Rotation 180:", rotation_180.shape)

print("Ukuran Horizontal Flip:", horizontal_flip.shape)
print("Ukuran Vertical Flip:", vertical_flip.shape)
