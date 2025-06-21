class PredictionException(Exception):
    """Exception raised when model prediction fails."""
    pass


class UserNotFoundException(Exception):
    """Exception raised when user ID is not found in the feature store."""
    pass 