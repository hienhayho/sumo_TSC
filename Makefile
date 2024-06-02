# Makefile

SHELL := /bin/bash
SCRIPTS_DIR := ./scripts
NUM_THREADS := 4
GPU := 1

.PHONY: train install

train:
	@echo "Ensuring all .sh files in the $(SCRIPTS_DIR) directory are executable..."
	@chmod +x $(SCRIPTS_DIR)/*.sh
	@echo "Running all .sh files in the $(SCRIPTS_DIR) directory..."
	@echo "Running all .sh files in the $(SCRIPTS_DIR) directory on GPU $(GPU) with $(NUM_THREADS) threads..."
	@ls $(SCRIPTS_DIR)/*.sh | xargs -n 1 -P $(NUM_THREADS) bash -c 'echo "Running $$0 on GPU $(GPU)"; $$0 $(GPU)'	
	done

install:
	@echo "Running install.sh..."
	@chmod +x ./install.sh && ./install.sh
