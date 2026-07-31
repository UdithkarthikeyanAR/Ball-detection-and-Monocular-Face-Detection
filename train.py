from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=0,          # Disable multiprocessing for now
        device=0,           # Use RTX 5050
        patience=20,
        cache=False,        # Better for 16 GB RAM
        pretrained=True,
        project="runs",
        name="tennis_ball"
    )

if __name__ == "__main__":
    main()