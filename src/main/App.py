from absl import app


def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass


# noinspection PyUnusedLocal
def main(*args):
    """
    :type args: parameters given from command line
    """
    tensorflow_shutup()
    from src.main.RecognizeElevatorElements import RecognizeElevatorElements

    application = RecognizeElevatorElements()
    while True:
        application.recognize_elevator_elements()


if __name__ == '__main__':
    app.run(main)
