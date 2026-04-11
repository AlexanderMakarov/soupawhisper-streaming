.PHONY: run run-stream run-no-stream test help service-stop service-start service-restart service-status service-logs record service-reinstall

run:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python dictate.py; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python dictate.py; \
	else \
		echo "Error: Neither uv nor poetry found. Please install one of them."; \
		exit 1; \
	fi

run-stream:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python dictate.py --streaming; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python dictate.py --streaming; \
	else \
		echo "Error: Neither uv nor poetry found. Please install one of them."; \
		exit 1; \
	fi

run-stream-verbose:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python dictate.py --streaming --verbose; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python dictate.py --streaming --verbose; \
	else \
		echo "Error: Neither uv nor poetry found. Please install one of them."; \
		exit 1; \
	fi

run-file:
	@if [ -z "$$F" ]; then \
		echo "Usage: make run-file F=<path/to/your_file.wav>"; \
		exit 1; \
	fi; \
	if command -v uv >/dev/null 2>&1; then \
		uv run python dictate.py --file "$$F"; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python dictate.py --file "$$F"; \
	else \
		echo "Error: Neither uv nor poetry found. Please install one of them."; \
		exit 1; \
	fi

run-no-stream:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python dictate.py --no-streaming; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python dictate.py --no-streaming; \
	else \
		echo "Error: Neither uv nor poetry found. Please install one of them."; \
		exit 1; \
	fi

test:
	@if command -v uv >/dev/null 2>&1; then \
		if [ ! -d .venv ]; then \
			uv venv .venv; \
		fi; \
		uv pip install --python .venv/bin/python -e . && \
		uv pip install --python .venv/bin/python "pytest>=8.0.0" "pytest-timeout>=2.0.0" && \
		.venv/bin/pytest dictate_tests.py; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry install && \
		poetry run pytest dictate_tests.py; \
	else \
		if [ ! -d .venv ]; then \
			python3 -m venv .venv; \
		fi; \
		.venv/bin/pip install -e . pytest && \
		.venv/bin/pytest dictate_tests.py; \
	fi

# Service targets: Linux = systemd, macOS = launchd (same label/paths as install.sh)
SOUPAWHISPER_PLIST = $(HOME)/Library/LaunchAgents/com.soupawhisper.dictate.plist
SOUPAWHISPER_LOG   = $(HOME)/Library/Logs/soupawhisper.log

service-stop:
	@if [ "$(shell uname -s)" = "Darwin" ]; then \
		launchctl unload "$(SOUPAWHISPER_PLIST)" 2>/dev/null && echo "Stopped soupawhisper" || echo "Service not running or not installed"; \
	else \
		command -v systemctl >/dev/null 2>&1 && systemctl --user stop soupawhisper || echo "Service not installed"; \
	fi

service-start:
	@if [ "$(shell uname -s)" = "Darwin" ]; then \
		if [ -f "$(SOUPAWHISPER_PLIST)" ]; then \
			launchctl load "$(SOUPAWHISPER_PLIST)" && echo "Started soupawhisper"; \
		else \
			echo "Service not installed. Run: make service-reinstall"; \
		fi; \
	else \
		command -v systemctl >/dev/null 2>&1 && systemctl --user start soupawhisper || echo "Service not installed"; \
	fi

service-restart:
	@if [ "$(shell uname -s)" = "Darwin" ]; then \
		if [ -f "$(SOUPAWHISPER_PLIST)" ]; then \
			launchctl unload "$(SOUPAWHISPER_PLIST)" 2>/dev/null; \
			launchctl load "$(SOUPAWHISPER_PLIST)" && echo "Restarted soupawhisper"; \
		else \
			echo "Service not installed. Run: make service-reinstall"; \
		fi; \
	else \
		command -v systemctl >/dev/null 2>&1 && systemctl --user restart soupawhisper || echo "Service not installed"; \
	fi

service-status:
	@if [ "$(shell uname -s)" = "Darwin" ]; then \
		if [ -f "$(SOUPAWHISPER_PLIST)" ]; then \
			launchctl list | grep -E "com.soupawhisper.dictate|PID" || echo "Service not loaded"; \
		else \
			echo "Service not installed. Run: make service-reinstall"; \
		fi; \
	else \
		command -v systemctl >/dev/null 2>&1 && systemctl --user status soupawhisper || echo "Service not installed"; \
	fi

service-logs:
	@if [ "$(shell uname -s)" = "Darwin" ]; then \
		if [ -f "$(SOUPAWHISPER_LOG)" ]; then \
			tail -f "$(SOUPAWHISPER_LOG)"; \
		else \
			echo "Log file not found. Start the service first: make service-start"; \
		fi; \
	else \
		command -v journalctl >/dev/null 2>&1 && journalctl --user -u soupawhisper -f || echo "journalctl not found"; \
	fi

record:
	@if command -v arecord >/dev/null 2>&1; then \
		FILENAME="/tmp/recording_$$(date +%Y%m%d_%H%M%S).wav"; \
		echo "Recording to $$FILENAME (Press Ctrl+C to stop)..."; \
		arecord -f S16_LE -r 16000 -c 1 -t wav "$$FILENAME"; \
		echo "Recording saved to $$FILENAME"; \
	else \
		echo "Error: arecord not found. Please install alsa-utils."; \
		exit 1; \
	fi

service-reinstall:
	@./install.sh --skip-deps --install-service
