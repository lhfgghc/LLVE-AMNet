class Logger:
    """Simple training logger that prints to console and optionally writes metrics."""

    def __init__(self, cfg, enabled=True):
        self.enabled = enabled

    def log(self, data, step=None):
        """Log metrics (no-op for console-only mode)."""
        pass

    def log_named_image(self, name, tensor, step=None):
        """Log a single image (no-op)."""
        pass

    def log_image_group(self, name, tensor_list, captions=None, step=None):
        """Log a group of images (no-op)."""
        pass

    def finish(self):
        """Finish logging session."""
        pass
