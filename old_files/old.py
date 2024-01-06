import argparse
import pickle
from collections import Counter
from pathlib import Path
import face_recognition
from PIL import Image, ImageDraw
import cv2
import os
import time

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
parser.add_argument("--live", action="store_true", help="Use live video feed for recognition")
args = parser.parse_args()

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        # wait_for_file(filepath)
        print(filepath)
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        # print(name)
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()


def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches.
    """
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]


def _display_face(draw, bounding_box, name):
    """
    Draws bounding boxes around faces, a caption area, and text captions.
    """
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )


def validate(model: str = "hog"):
    """
    Runs recognize_faces on a set of images with known faces to validate
    known encodings.
    """
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )




def recognize_faces_live(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH,
                         resize_factor: float = 0.5, skip_frames: int = 2) -> None:
    """
    Recognize faces from live video feed using the webcam.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    video_capture = cv2.VideoCapture(0)  # Use the default webcam (you can change the index if you have multiple)

    frame_count = 0

    while True:
        ret, frame = video_capture.read()

        # Resize frame for faster processing
        frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
        frame_count += 1

        if frame_count % skip_frames == 0:
            input_face_locations = face_recognition.face_locations(frame, model=model)
            input_face_encodings = face_recognition.face_encodings(frame, input_face_locations)

            for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
                name = _recognize_face(unknown_encoding, loaded_encodings)
                if not name:
                    name = "Unknown"
                # _display_face_cv(frame, bounding_box, name)
                # better_display_face(frame, bounding_box, name)
                best_display_face(frame, bounding_box, name)
                print(name)

            cv2.imshow("Video Feed", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# def _display_face_cv(frame, bounding_box, name):
#     top, right, bottom, left = bounding_box
#     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#     font = cv2.FONT_HERSHEY_DUPLEX
#     cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

# def better_display_face(frame, bounding_box, name):
#     top, right, bottom, left = bounding_box
#     frame  = cv2.rectangle(frame ,(left,top),(right,bottom),(255,0,0),thickness = 4)
#     frame  = cv2.putText(frame , name, (left - 5 ,top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def best_display_face(frame, bounding_box, name):
    top, right, bottom, left = bounding_box

    frame  = cv2.rectangle(frame ,(int(left), int(top)),(int(right), int(bottom)), color = (255,0,0),thickness = 4)

    frame  = cv2.putText(frame, name ,(int(left)-5, int(top)-5),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = 0.5,thickness=2, color = (255,255,0))

# Update your if __name__ == "__main__" block
if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
    if args.live:
        recognize_faces_live(model=args.m)

