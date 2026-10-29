import os
from PIL import Image, ImageDraw


def generate_demonstration_samples():
    sample_dir = "./data/samples"
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Fish / Aquatic sample (for Feature #588 and #2077)
    fish_img = Image.new("RGB", (224, 224), color=(30, 80, 140))  # Water background
    draw = ImageDraw.Draw(fish_img)
    # Background bubbles / water caustics
    for y in range(0, 224, 28):
        draw.line([0, y, 224, y + 10], fill=(45, 110, 180), width=2)
    # Fish body (goldfish / tench color)
    draw.polygon([(40, 112), (150, 70), (180, 112), (150, 154)], fill=(235, 120, 20))
    # Tail fin
    draw.polygon([(40, 112), (15, 75), (15, 149)], fill=(210, 85, 15))
    # Dorsal fin
    draw.polygon([(110, 70), (135, 40), (145, 70)], fill=(210, 85, 15))
    # Scales pattern
    for sx in range(70, 160, 14):
        for sy in range(85, 140, 14):
            draw.arc([sx - 8, sy - 8, sx + 8, sy + 8], start=180, end=360, fill=(255, 180, 50), width=2)
    # Eye
    draw.ellipse([160, 100, 172, 112], fill=(255, 255, 255))
    draw.ellipse([165, 103, 170, 108], fill=(10, 10, 20))
    fish_img.save(os.path.join(sample_dir, "sample_fish.png"))

    # 2. English Springer Spaniel sample (for Feature #1453)
    springer_img = Image.new("RGB", (224, 224), color=(220, 225, 215))  # Park grass background
    draw = ImageDraw.Draw(springer_img)
    # Dog head silhouette
    draw.ellipse([60, 50, 164, 154], fill=(245, 245, 245))  # White blaze & muzzle
    # Liver/dark ear patches characteristic of Springer Spaniel
    draw.polygon([(60, 60), (35, 130), (75, 140)], fill=(65, 38, 22))  # Left ear
    draw.polygon([(164, 60), (189, 130), (149, 140)], fill=(65, 38, 22))  # Right ear
    draw.ellipse([65, 55, 95, 85], fill=(65, 38, 22))  # Eye patch left
    draw.ellipse([129, 55, 159, 85], fill=(65, 38, 22))  # Eye patch right
    # Eyes & muzzle
    draw.ellipse([78, 68, 88, 78], fill=(20, 15, 10))
    draw.ellipse([136, 68, 146, 78], fill=(20, 15, 10))
    draw.ellipse([98, 110, 126, 134], fill=(25, 20, 20))  # Nose
    # Feathered coat texture spots
    for sp in [(80, 150), (110, 160), (140, 155), (95, 180), (130, 185)]:
        draw.ellipse([sp[0] - 6, sp[1] - 4, sp[0] + 6, sp[1] + 4], fill=(65, 38, 22))
    springer_img.save(os.path.join(sample_dir, "sample_springer.png"))

    # 3. Golden Retriever sample
    retriever_img = Image.new("RGB", (224, 224), color=(210, 175, 130))
    draw = ImageDraw.Draw(retriever_img)
    for x in range(30, 194, 4):
        for y in range(40, 180, 6):
            draw.line([x, y, x + 2, y + 5], fill=(139, 90, 43), width=2)
    draw.polygon([(40, 50), (20, 110), (70, 90)], fill=(110, 65, 30))
    draw.polygon([(184, 50), (204, 110), (154, 90)], fill=(110, 65, 30))
    draw.ellipse([90, 120, 134, 150], fill=(40, 25, 20))
    draw.ellipse([105, 125, 119, 137], fill=(10, 10, 10))
    retriever_img.save(os.path.join(sample_dir, "sample_retriever.png"))

    print("Demonstration sample images saved to ./data/samples/")


if __name__ == "__main__":
    generate_demonstration_samples()

