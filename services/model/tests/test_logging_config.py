import logging


def test_setup_logging_has_no_always_on_file_handler_and_filters_console(capsys, tmp_path, monkeypatch):
    from model_service.core.logging_config import OPS_LOGGER_NAME, setup_logging

    monkeypatch.setenv("MODEL__LOGGING__LOG_DIR", str(tmp_path))
    root_logger = logging.getLogger()
    old_handlers = list(root_logger.handlers)
    old_level = root_logger.level

    try:
        setup_logging("INFO")

        assert not any(
            isinstance(handler, logging.FileHandler)
            for handler in root_logger.handlers
        )

        logging.getLogger("model_service.video").info("detailed trace line")
        logging.getLogger(OPS_LOGGER_NAME).info("operator summary line")
        output = capsys.readouterr()

        combined = output.out + output.err
        assert "operator summary line" in combined
        assert "detailed trace line" not in combined
        assert list(tmp_path.glob("*")) == []
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()
        for handler in old_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(old_level)
