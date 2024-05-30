# Makefile

SHELL := /bin/bash
SCRIPTS_DIR := ./scripts

.PHONY: train install

train:
	@echo "Ensuring all .sh files in the $(SCRIPTS_DIR) directory are executable..."
	@chmod +x $(SCRIPTS_DIR)/*.sh
	@echo "Running all .sh files in the $(SCRIPTS_DIR) directory..."
	@for script in $(SCRIPTS_DIR)/*.sh; do \
		echo "Running $$script"; \
		$$script 0; \
	done

install:
	@echo "Running install.sh..."
	@chmod +x ./install.sh && ./install.sh