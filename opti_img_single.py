from PIL import Image

image = Image.open("code.png")
image.thumbnail((256, 256))
image.save("codeOpti.png")
