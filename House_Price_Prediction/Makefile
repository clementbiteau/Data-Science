PROJECT_NAME := $(notdir $(CURDIR))
DATE := $(shell date "+%Y-%m-%d")
AUTHOR := Clément Biteau

define HEADER
############################################################

$(PROJECT_NAME) by $(AUTHOR)

$(DATE)

Source: Kaggle House Prices Prediction

############################################################
endef
$(info $(HEADER))

PYTHON := python3

MAIN := main.py

REPORTS_DIR := scripts/models/reports

all: run

run: $(MAIN)
	@echo "Running house prices prediction pipeline..."
	$(PYTHON) $(MAIN)

prepare:
	@echo "Ensuring output directories exist..."
	mkdir -p $(REPORTS_DIR)

clean:
	@echo "Cleaning generated files..."
	rm -f $(REPORTS_DIR)/*.png

rebuild: clean prepare run

help:
	@echo "Usage:"
	@echo "  make           - Run the full pipeline"
	@echo "  make run       - Execute main.py"
	@echo "  make prepare   - Create required directories"
	@echo "  make clean     - Remove generated reports list"
	@echo "  make rebuild   - Clean and rerun everything"
