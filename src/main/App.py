from src.main.RecognizeElevatorElements import RecognizeElevatorElements
from absl import app


def main(*args):
    application = RecognizeElevatorElements()
    while True:
        application.recognize_elevator_elements()


if __name__ == '__main__':
    app.run(main)
