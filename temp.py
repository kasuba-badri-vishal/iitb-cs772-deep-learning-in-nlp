import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

data = "गुलाबी कार्ड पोनी छोटी लड़कियों के लिए नॉक-जीन बेचती है।\n पिंजरे में बबूआ आपको बताता है कि आप बड़े होने पर क्या होंगे।"

blank_image = Image.new('RGB', (1024, 512))
font = ImageFont.truetype('data/font.ttf', size=40, encoding='unic')
draw = ImageDraw.Draw(blank_image)
draw.text((10, 10), data, font=font, fill=(255, 0, 0))
cv2_image = cv2.cvtColor(np.array(blank_image), cv2.COLOR_RGB2BGR)
cv2.imwrite("temp.jpg", cv2_image)