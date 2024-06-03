import time

class Delay:
  """
  A simple class to introduce a delay in your program.
  """
  def __init__(self, delay_time_in_seconds):
    """
    Initialize the Delay class with a specified delay time.

    Args:
      delay_time_in_seconds: The amount of time to delay in seconds (float).
    """
    self.delay_time = delay_time_in_seconds

  def wait(self):
    """
    Causes the program to wait for the specified delay time.

    This function uses the `time.sleep` function from the standard library.
    """
    
    time.sleep(self.delay_time)
