from ultralytics import YOLO

if __name__ == "__main__":

    # Load your previously trained model
    model = YOLO("runs/detect/runs/tennis_ball-4/weights/best.pt")

    # Fine-tune on the new multi-class dataset
    model.train(
        data="dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=10,
        cache=True,
        amp=True,
        project="runs",
        name="multiball_v1"
    )