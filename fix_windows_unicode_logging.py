"""
Fix for Windows Unicode logging issues
"""

import os
import sys
import logging


def fix_unicode_logging():
    """Fix Unicode encoding issues in Windows logging"""

    # Set UTF-8 encoding for the environment
    if os.name == "nt":  # Windows
        # Set environment variables for UTF-8
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["PYTHONUTF8"] = "1"

        # Reconfigure standard streams with UTF-8
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")

    # Set up proper logging formatter that avoids problematic Unicode characters
    class SafeFormatter(logging.Formatter):
        """Safe formatter that replaces problematic Unicode characters"""

        def format(self, record):
            # Replace problematic Unicode characters with safe alternatives
            if hasattr(record, "msg"):
                if isinstance(record.msg, str):
                    # Replace common problematic characters
                    record.msg = record.msg.replace("✅", "[SUCCESS]")
                    record.msg = record.msg.replace("❌", "[ERROR]")
                    record.msg = record.msg.replace("⚠️", "[WARNING]")
                    record.msg = record.msg.replace("🎯", "[TARGET]")
                    record.msg = record.msg.replace("🔍", "[SEARCH]")
                    record.msg = record.msg.replace("📊", "[DATA]")
                    record.msg = record.msg.replace("💾", "[SAVE]")
                    record.msg = record.msg.replace("🚀", "[LAUNCH]")

            try:
                return super().format(record)
            except UnicodeEncodeError:
                # Fallback to ASCII if Unicode fails
                record.msg = str(record.msg).encode("ascii", "replace").decode("ascii")
                return super().format(record)

    # Apply safe formatter to existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            if hasattr(handler, "setFormatter"):
                safe_formatter = SafeFormatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(safe_formatter)


if __name__ == "__main__":
    fix_unicode_logging()
    print("Unicode logging fix applied successfully!")
