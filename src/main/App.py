from src.main.RecognizeElevatorElements import RecognizeElevatorElements
from absl import app


# noinspection PyUnusedLocal
def main(*args):
    """
    :type args: parameters given from command line
    """
    application = RecognizeElevatorElements()
    while True:
        application.recognize_elevator_elements()


if __name__ == '__main__':
    app.run(main)
