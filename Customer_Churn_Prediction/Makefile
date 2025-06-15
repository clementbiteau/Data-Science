PROJECT_NAME := $(notdir $(CURDIR))
DATE := $(shell date "+%Y-%m-%d")
AUTHOR := Clément Biteau

define HEADER
############################################################

$(PROJECT_NAME) by $(AUTHOR)

$(DATE)

Source: Kaggle Telco Customer Churn Prediction

############################################################
endef
$(info $(HEADER))

PYTHON := python3

MAIN := main.py

TOP_CHURNERS_DIR := data/top_churners
REPORTS_DIR := scripts/models/reports

all: run

run: $(MAIN)
	@echo "Running churn prediction pipeline..."
	$(PYTHON) $(MAIN)

prepare:
	@echo "Ensuring output directories exist..."
	mkdir -p $(TOP_CHURNERS_DIR)
	mkdir -p $(REPORTS_DIR)

clean:
	@echo "Cleaning generated files..."
	rm -f $(REPORTS_DIR)/*.png
	rm -f $(TOP_CHURNERS_DIR)/*.csv

rebuild: clean prepare run

help:
	@echo "Usage:"
	@echo "  make           - Run the full pipeline"
	@echo "  make run       - Execute main.py"
	@echo "  make prepare   - Create required directories"
	@echo "  make clean     - Remove generated reports and churners list"
	@echo "  make rebuild   - Clean and rerun everything"
