all: setup run

run: venv/bin/activate
	./venv/bin/python3 src/pipeline.py

setup: requirements.txt
	virtualenv venv
	./venv/bin/pip install -r requirements.txt
	
clean:
	rm -rf src/__pycache__
	rm -rf venv