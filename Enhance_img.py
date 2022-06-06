from PIL import Image
from PIL import ImageFilter

# Uploading 5 Images, and showing them
stack = []
for i in range(5):
    print('Image is loading...')
    img = Image.open(f'img_{i+1}.jpg')
    stack.append(img)
# print(stack)
for j in range(5):
    stack[j].show()

# Converting grayscale image to binary image
# stack_Binary = []
# for i in range(5):
#     print('Converting grayscale image to binary image is loading...')
#     img_gray = stack[i].convert('L')
#     binary_img = img_gray.point(lambda x: 0 if x < 128 else 255, '1')
#     stack_Binary.append(binary_img)

# # Removing a background of an image
# stack_BG = []
# for j in range(5):
#     print('Removing a background of an image is loading...')
#     img_BG = stack[i].convert('RGBA')
#     for item in img_BG.getdata():
#         if item[:3] == (255, 255, 255):
#             stack_BG.append((255, 255, 255, 0))
#         else:
#             stack_BG.append(item)
#     img_BG.putdata(stack_BG)
# for k in range(5):
#     stack_BG[k].show()
# for m in range(5):
#     new_img_without_BG = stack_BG[m].save(f'img_{i+1}_without_BG.jpg')

# Enhancing images by soomthing
stack_Smooth = []
for i in range(5):
    print('Smoothing an image is loading...')
    smooth_img = stack[i].filter(ImageFilter.SMOOTH)
    stack_Smooth.append(smooth_img)
for j in range(5):
    stack_Smooth[j].show()
for k in range(5):
    new_imgSmooth = stack_Smooth[j].save(f'img_{i+1}_smooth.jpg')

# Enhancing images by sharping
stack_Sharp = []
for i in range(5):
    print('Sharping an image is loading...')
    sharpen_img = stack[i].filter(ImageFilter.SHARPEN)
    stack_Sharp.append(sharpen_img)
for j in range(5):
    stack_Sharp[j].show()
for k in range(5):
    new_imgSharpe = stack_Sharp[j].save(f'img_{i+1}_sharpen.jpg')

# Detecting edge of images
stack_Edge = []
for i in range(5):
    print('Detecting an edge of image is loading...')
    find_edges = stack[i].filter(ImageFilter.FIND_EDGES)
    stack_Edge.append(find_edges)
for j in range(5):
    stack_Edge[j].show()
for k in range(5):
    new_imgEdge = stack_Edge[j].save(f'img_{i+1}_edge.jpg')

# Coverting orignal images to gray images
stack_Gray = []
for i in range(5):
    print('Graying an image is loading...')
    imgGray = stack[i].convert('L')
    stack_Gray.append(imgGray)
for j in range(5):
    stack_Gray[j].show()
for k in range(5):
    new_imgGray = stack_Gray[j].save(f'img_{i+1}_gray.jpg')
