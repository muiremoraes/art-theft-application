# Art Watch
Web application to help protect digital artists from theft and AI scrapers.

## Technqiues
- Watermarking - DCT, LSB, visisble
- Image comparison - SIFT, ORB
- Reverse image search
- FGSM AI defence



## Setup
- Python 3 and above
- pip

```bash
cd Art-Watch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

- create a file called `config.py`
```python
SERPAPI_KEY = "key"
IMAGE_BB_KEY = "key"
GMAIL_ID= "email"
GMAIL_PASSWORD="pass"
SECRET_KEY ="key"
JWT_SECRET_KEY="key"
```


### Run
```
venv\Scripts\activate
python main.py
```

### References and sources 
[references.md](references.md)
