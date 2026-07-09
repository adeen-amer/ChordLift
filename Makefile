.PHONY: eval test eval-gold-dev eval-gold-test eval-gold-all phase13-v49 phase-v50 build-gold-bundle verify-gold-audio

PYTHON ?= $(shell if [ -x backend/.venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
BACKEND = backend

eval:
	cd $(BACKEND) && CHORD_ENGINE=ml CHORD_ENGINE_STRICT=1 CHORD_ML_MODEL=chordia \
		$(PYTHON) eval_gold_mir.py --no-cache --split dev --require-audio-identity --check-baseline && \
		CHORD_ENGINE=ml CHORD_ENGINE_STRICT=1 CHORD_ML_MODEL=chordia \
		$(PYTHON) eval_gold_mir.py --no-cache --split test --require-audio-identity --check-baseline

eval-gold-dev:
	cd $(BACKEND) && CHORD_ENGINE=ml CHORD_ENGINE_STRICT=1 CHORD_ML_MODEL=chordia \
		$(PYTHON) eval_gold_mir.py --no-cache --split dev --require-audio-identity

eval-gold-test:
	cd $(BACKEND) && CHORD_ENGINE=ml CHORD_ENGINE_STRICT=1 CHORD_ML_MODEL=chordia \
		$(PYTHON) eval_gold_mir.py --no-cache --split test --require-audio-identity

eval-gold-all:
	cd $(BACKEND) && CHORD_ENGINE=ml CHORD_ENGINE_STRICT=1 CHORD_ML_MODEL=chordia \
		$(PYTHON) eval_gold_mir.py --no-cache --require-audio-identity

phase13-v49:
	cd $(BACKEND) && CHORD_ENGINE=ml CHORD_ENGINE_STRICT=1 CHORD_ML_MODEL=chordia \
		$(PYTHON) scripts/rebaseline_v49.py

phase-v50:
	cd $(BACKEND) && \
		$(PYTHON) scripts/rebaseline_v50.py --split both --select confidence

verify-gold-audio:
	cd $(BACKEND) && $(PYTHON) scripts/verify_gold_audio.py

build-gold-bundle:
	cd $(BACKEND) && $(PYTHON) scripts/build_gold_audio_bundle.py --print-sha256

test:
	cd $(BACKEND) && $(PYTHON) -m pytest tests/ -q --ignore=tests/fixtures
